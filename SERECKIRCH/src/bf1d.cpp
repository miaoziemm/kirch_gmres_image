#include "../include/bf1d.h"

#include <map>
#include <memory>
#include <mutex>
#include <tuple>



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef BF_PHASE_RES_TOL
#define BF_PHASE_RES_TOL 2.0f
#endif


struct BFC {
    float r;
    float i;
};

static inline BFC c_make(float r, float i) { BFC z; z.r = r; z.i = i; return z; }
static inline BFC c_add(BFC a, BFC b) { return c_make(a.r + b.r, a.i + b.i); }
static inline void c_acc(BFC &a, BFC b) { a.r += b.r; a.i += b.i; }
static inline BFC c_mul(BFC a, BFC b) { return c_make(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r); }
static inline BFC c_scale(BFC a, float s) { return c_make(a.r*s, a.i*s); }
static inline BFC c_exp(double ph)
{
    const float x = static_cast<float>(ph);
    float sine = 0.0f;
    float cosine = 0.0f;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_sincosf(x, &sine, &cosine);
#else
    cosine = std::cos(x);
    sine = std::sin(x);
#endif
    return c_make(cosine, sine);
}

static void bf_die(const char *msg)
{
    std::fprintf(stderr, "bf1d_strict_butterfly error: %s\n", msg);
    std::exit(EXIT_FAILURE);
}

static inline size_t mat_index(int row, int col, int n)
{
    return (size_t)row * (size_t)n + (size_t)col;
}

static int is_power_of_two(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

static int int_log2_exact(int x)
{
    if (!is_power_of_two(x)) bf_die("argument is not a power of two.");
    int L = 0;
    while ((1 << L) < x) L++;
    return L;
}

static int next_legal_size(int n, int leaf_n)
{
    if (n <= 0 || leaf_n <= 0) bf_die("invalid size.");
    int m = leaf_n;
    while (m < n) m <<= 1;
    return m;
}

static int panel_size_from_levels(int leaf_n, int panel_levels)
{
    if (panel_levels < 0) panel_levels = 2;
    int n = leaf_n;
    for (int i = 0; i < panel_levels; ++i) n <<= 1;
    return n;
}

struct BFBox1D {
    int i1;
    int i2;
    double x0;
    double width;
    std::vector<double> cheb;
};

static void build_cheb_std(int p, std::vector<double> &cheb_std)
{
    cheb_std.resize((size_t)p);
    for (int k = 0; k < p; k++) {
        cheb_std[(size_t)k] = std::cos(((2.0 * (double)(k + 1) - 1.0) * M_PI) / (2.0 * (double)p));
    }
}

static void build_boxes(int n, int leaf_n, int p, std::vector< std::vector<BFBox1D> > &levels)
{
    int L = int_log2_exact(n / leaf_n);
    std::vector<double> cheb_std;
    build_cheb_std(p, cheb_std);
    levels.clear();
    levels.resize((size_t)L + 1);

    for (int lev = 0; lev <= L; ++lev) {
        int nbox = 1 << lev;
        int box_n = n / nbox;
        levels[(size_t)lev].resize((size_t)nbox);
        for (int b = 0; b < nbox; ++b) {
            int i1 = b * box_n;
            int i2 = (b + 1) * box_n - 1;
            double xL = (double)i1;
            double xR = (double)i2;
            double x0 = 0.5 * (xL + xR);
            double width = xR - xL;
            BFBox1D &box = levels[(size_t)lev][(size_t)b];
            box.i1 = i1;
            box.i2 = i2;
            box.x0 = x0;
            box.width = width;
            box.cheb.resize((size_t)p);
            for (int it = 0; it < p; ++it) {
                box.cheb[(size_t)it] = x0 + 0.5 * width * cheb_std[(size_t)it];
            }
        }
    }
}

static void bary_weights(const std::vector<double> &nodes, std::vector<double> &w)
{
    int p = (int)nodes.size();
    w.assign((size_t)p, 1.0);
    for (int j = 0; j < p; ++j) {
        double v = 1.0;
        for (int k = 0; k < p; ++k) {
            if (k != j) v /= (nodes[(size_t)j] - nodes[(size_t)k]);
        }
        w[(size_t)j] = v;
    }
}

static void lagrange_basis_all(double x,
                               const std::vector<double> &nodes,
                               const std::vector<double> &w,
                               std::vector<double> &L)
{
    int p = (int)nodes.size();
    const double tol = 1e-13;
    L.assign((size_t)p, 0.0);
    int hit = -1;
    for (int j = 0; j < p; ++j) {
        if (std::fabs(x - nodes[(size_t)j]) < tol) {
            hit = j;
            break;
        }
    }
    if (hit >= 0) {
        L[(size_t)hit] = 1.0;
        return;
    }
    double denom = 0.0;
    for (int j = 0; j < p; ++j) {
        L[(size_t)j] = w[(size_t)j] / (x - nodes[(size_t)j]);
        denom += L[(size_t)j];
    }
    for (int j = 0; j < p; ++j) L[(size_t)j] /= denom;
}

static void build_lagrange_matrix(const std::vector<double> &points,
                                  const std::vector<double> &nodes,
                                  std::vector<double> &Lmat)
{
    int m = (int)points.size();
    int p = (int)nodes.size();
    std::vector<double> w, basis;
    bary_weights(nodes, w);
    Lmat.assign((size_t)m * (size_t)p, 0.0);
    for (int k = 0; k < m; ++k) {
        lagrange_basis_all(points[(size_t)k], nodes, w, basis);
        for (int it = 0; it < p; ++it) {
            Lmat[(size_t)k * (size_t)p + (size_t)it] = basis[(size_t)it];
        }
    }
}

static double interp_real_rowcol(const float *A, int n, double source_coord, double target_coord)
{
    const double tol = 1e-12;
    double x = source_coord;
    double z = target_coord;
    if (x < -tol || x > (double)(n - 1) + tol || z < -tol || z > (double)(n - 1) + tol) {
        bf_die("real interpolation query is outside matrix.");
    }
    if (x < 0.0) x = 0.0;
    if (x > (double)(n - 1)) x = (double)(n - 1);
    if (z < 0.0) z = 0.0;
    if (z > (double)(n - 1)) z = (double)(n - 1);

    int j_round = (int)llround(x);
    int i_round = (int)llround(z);
    if (std::fabs(x - (double)j_round) < tol && std::fabs(z - (double)i_round) < tol) {
        return (double)A[mat_index(i_round, j_round, n)];
    }

    int j1 = (int)std::floor(x);
    int i1 = (int)std::floor(z);
    if (j1 >= n - 1) j1 = n - 2;
    if (i1 >= n - 1) i1 = n - 2;
    if (j1 < 0) j1 = 0;
    if (i1 < 0) i1 = 0;
    int j2 = j1 + 1;
    int i2 = i1 + 1;
    double tx = x - (double)j1;
    double tz = z - (double)i1;
    double Q11 = (double)A[mat_index(i1, j1, n)];
    double Q21 = (double)A[mat_index(i1, j2, n)];
    double Q12 = (double)A[mat_index(i2, j1, n)];
    double Q22 = (double)A[mat_index(i2, j2, n)];
    return (1.0 - tx) * (1.0 - tz) * Q11
         + tx         * (1.0 - tz) * Q21
         + (1.0 - tx) * tz         * Q12
         + tx         * tz         * Q22;
}

static BFC interp_complex_rowcol(const BFC *A, int n, double source_coord, double target_coord)
{
    const double tol = 1e-12;
    double x = source_coord;
    double z = target_coord;
    if (x < -tol || x > (double)(n - 1) + tol || z < -tol || z > (double)(n - 1) + tol) {
        bf_die("complex interpolation query is outside matrix.");
    }
    if (x < 0.0) x = 0.0;
    if (x > (double)(n - 1)) x = (double)(n - 1);
    if (z < 0.0) z = 0.0;
    if (z > (double)(n - 1)) z = (double)(n - 1);
    int j_round = (int)llround(x);
    int i_round = (int)llround(z);
    if (std::fabs(x - (double)j_round) < tol && std::fabs(z - (double)i_round) < tol) {
        return A[mat_index(i_round, j_round, n)];
    }
    int j1 = (int)std::floor(x);
    int i1 = (int)std::floor(z);
    if (j1 >= n - 1) j1 = n - 2;
    if (i1 >= n - 1) i1 = n - 2;
    if (j1 < 0) j1 = 0;
    if (i1 < 0) i1 = 0;
    int j2 = j1 + 1;
    int i2 = i1 + 1;
    double tx = x - (double)j1;
    double tz = z - (double)i1;
    double w11 = (1.0 - tx) * (1.0 - tz);
    double w21 = tx         * (1.0 - tz);
    double w12 = (1.0 - tx) * tz;
    double w22 = tx         * tz;
    BFC q11 = A[mat_index(i1, j1, n)];
    BFC q21 = A[mat_index(i1, j2, n)];
    BFC q12 = A[mat_index(i2, j1, n)];
    BFC q22 = A[mat_index(i2, j2, n)];
    return c_make((float)(w11*q11.r + w21*q21.r + w12*q12.r + w22*q22.r),
                  (float)(w11*q11.i + w21*q21.i + w12*q12.i + w22*q22.i));
}

static inline size_t sig_idx(int ibox_s, int ibox_r, int it, int nbox_r, int p)
{
    return ((size_t)ibox_s * (size_t)nbox_r + (size_t)ibox_r) * (size_t)p + (size_t)it;
}


struct R1Geometry {
    int lv;
    int nbox_r_now;
    int nbox_s_now;
    int nbox_r_prev;
    int nbox_s_prev;
    std::vector<double> L_child_to_parent;
};

struct R2Geometry {
    int lv;
    int nbox_r_now;
    int nbox_s_now;
    int nbox_r_prev;
    int nbox_s_prev;
    std::vector<double> L_target;
};

struct BFGeometryTemplate {
    int n;
    int p;
    int leaf_n;
    int L;
    std::vector< std::vector<BFBox1D> > TR;
    std::vector< std::vector<BFBox1D> > TS;
    std::vector<double> init_L;
    int lv_max;
    std::vector<R1Geometry> r1;
    int nbox_r_mid;
    int nbox_s_mid;
    std::vector<R2Geometry> r2;
    std::vector<double> term_L;
};

static std::shared_ptr<const BFGeometryTemplate>
get_geometry_template(int n, int p, int leaf_n)
{
    using Key = std::tuple<int, int, int>;
    static std::mutex cache_mutex;
    static std::map<Key, std::shared_ptr<const BFGeometryTemplate>> cache;
    /*
     * A segmented factor normally contains hundreds or thousands of strict
     * panels with identical geometry.  Taking the process-wide mutex for
     * every panel used to serialize otherwise independent factor builds.
     * Keep a reference in each builder thread after its first global lookup;
     * shared_ptr keeps the object alive and makes the common path lock-free.
     */
    static thread_local std::map<
        Key, std::shared_ptr<const BFGeometryTemplate>> local_cache;

    const Key key(n, p, leaf_n);
    const auto local_found = local_cache.find(key);
    if (local_found != local_cache.end()) return local_found->second;

    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) {
        local_cache.emplace(key, found->second);
        return found->second;
    }

    auto G = std::make_shared<BFGeometryTemplate>();
    G->n = n;
    G->p = p;
    G->leaf_n = leaf_n;
    G->L = int_log2_exact(n / leaf_n);
    build_boxes(n, leaf_n, p, G->TR);
    build_boxes(n, leaf_n, p, G->TS);

    const int L = G->L;
    const int nbox_s_leaf = 1 << L;
    G->init_L.assign(
        (size_t)nbox_s_leaf * (size_t)leaf_n * (size_t)p, 0.0);
    for (int ibox_s = 0; ibox_s < nbox_s_leaf; ++ibox_s) {
        const BFBox1D &B = G->TS[(size_t)L][(size_t)ibox_s];
        std::vector<double> points((size_t)leaf_n);
        for (int k = 0; k < leaf_n; ++k) {
            points[(size_t)k] = (double)(B.i1 + k);
        }
        std::vector<double> Lg;
        build_lagrange_matrix(points, B.cheb, Lg);
        for (int k = 0; k < leaf_n; ++k) {
            for (int it = 0; it < p; ++it) {
                G->init_L[
                    ((size_t)ibox_s * (size_t)leaf_n + (size_t)k) *
                        (size_t)p +
                    (size_t)it] =
                    Lg[(size_t)k * (size_t)p + (size_t)it];
            }
        }
    }

    G->lv_max = L / 2;
    for (int lv = 1; lv <= G->lv_max; ++lv) {
        R1Geometry S;
        S.lv = lv;
        S.nbox_r_now = 1 << lv;
        S.nbox_s_now = 1 << (L - lv);
        S.nbox_r_prev = 1 << (lv - 1);
        S.nbox_s_prev = 1 << (L - lv + 1);
        S.L_child_to_parent.assign(
            (size_t)S.nbox_s_now * 2u * (size_t)p * (size_t)p,
            0.0);
        for (int ibox_s = 0; ibox_s < S.nbox_s_now; ++ibox_s) {
            const BFBox1D &B =
                G->TS[(size_t)(L - lv)][(size_t)ibox_s];
            for (int child = 0; child < 2; ++child) {
                const int ibox_s_child = 2 * ibox_s + child;
                const BFBox1D &Bc =
                    G->TS[(size_t)(L - lv + 1)]
                         [(size_t)ibox_s_child];
                std::vector<double> Lmat;
                build_lagrange_matrix(Bc.cheb, B.cheb, Lmat);
                for (int jt = 0; jt < p; ++jt) {
                    for (int it = 0; it < p; ++it) {
                        S.L_child_to_parent[
                            (((size_t)ibox_s * 2u + (size_t)child) *
                                 (size_t)p +
                             (size_t)jt) *
                                (size_t)p +
                            (size_t)it] =
                            Lmat[(size_t)jt * (size_t)p +
                                 (size_t)it];
                    }
                }
            }
        }
        G->r1.push_back(std::move(S));
    }

    G->nbox_r_mid = 1 << G->lv_max;
    G->nbox_s_mid = 1 << (L - G->lv_max);

    for (int lv = G->lv_max + 1; lv <= L; ++lv) {
        R2Geometry S;
        S.lv = lv;
        S.nbox_r_now = 1 << lv;
        S.nbox_s_now = 1 << (L - lv);
        S.nbox_r_prev = 1 << (lv - 1);
        S.nbox_s_prev = 1 << (L - lv + 1);
        S.L_target.assign(
            (size_t)S.nbox_r_now * (size_t)p * (size_t)p,
            0.0);
        for (int ibox_r = 0; ibox_r < S.nbox_r_now; ++ibox_r) {
            const BFBox1D &A = G->TR[(size_t)lv][(size_t)ibox_r];
            const int ibox_r_parent = ibox_r / 2;
            const BFBox1D &Ap =
                G->TR[(size_t)(lv - 1)][(size_t)ibox_r_parent];
            std::vector<double> Lmat;
            build_lagrange_matrix(A.cheb, Ap.cheb, Lmat);
            for (int it = 0; it < p; ++it) {
                for (int jt = 0; jt < p; ++jt) {
                    S.L_target[
                        ((size_t)ibox_r * (size_t)p + (size_t)it) *
                            (size_t)p +
                        (size_t)jt] =
                        Lmat[(size_t)it * (size_t)p + (size_t)jt];
                }
            }
        }
        G->r2.push_back(std::move(S));
    }

    const int nbox_r_leaf = 1 << L;
    G->term_L.assign(
        (size_t)nbox_r_leaf * (size_t)leaf_n * (size_t)p,
        0.0);
    for (int ibox_r = 0; ibox_r < nbox_r_leaf; ++ibox_r) {
        const BFBox1D &A = G->TR[(size_t)L][(size_t)ibox_r];
        std::vector<double> points((size_t)leaf_n);
        for (int k = 0; k < leaf_n; ++k) {
            points[(size_t)k] = (double)(A.i1 + k);
        }
        std::vector<double> Lg;
        build_lagrange_matrix(points, A.cheb, Lg);
        for (int k = 0; k < leaf_n; ++k) {
            for (int it = 0; it < p; ++it) {
                G->term_L[
                    ((size_t)ibox_r * (size_t)leaf_n + (size_t)k) *
                        (size_t)p +
                    (size_t)it] =
                    Lg[(size_t)k * (size_t)p + (size_t)it];
            }
        }
    }

    const std::shared_ptr<const BFGeometryTemplate> result = G;
    cache.emplace(key, result);
    local_cache.emplace(key, result);
    return result;
}

struct R1Stage {
    int lv;
    int nbox_r_now;
    int nbox_s_now;
    int nbox_r_prev;
    int nbox_s_prev;
    const std::vector<double>* L_child_to_parent; /* shared geometry */
    std::vector<BFC> phase_child;          /* [ibox_r][ibox_s_child][jt] */
    std::vector<BFC> phase_parent;         /* [ibox_r][ibox_s][it] */
};

struct R2Stage {
    int lv;
    int nbox_r_now;
    int nbox_s_now;
    int nbox_r_prev;
    int nbox_s_prev;
    const std::vector<double>* L_target; /* shared geometry */
    std::vector<BFC> phase_rp;     /* [ibox_r][ibox_s_child][jt] */
    std::vector<BFC> phase_r;      /* [ibox_r][ibox_s_child][it] */
};

struct BFStrictFactor {
    int n;
    int p;
    int leaf_n;
    int L;
    double alpha;
    std::shared_ptr<const BFGeometryTemplate> geometry;

    std::vector<BFC> init_phase_s;   /* [ibox_s][local] */
    std::vector<BFC> init_phase_node;/* [ibox_s][it] */

    int lv_max;
    std::vector<R1Stage> r1;

    int nbox_r_mid;
    int nbox_s_mid;
    std::vector<BFC> switch_kernel;  /* [ibox_s][ibox_r][it][jt] */

    std::vector<R2Stage> r2;

    std::vector<BFC> term_phase_node; /* [ibox_r][it] */
    std::vector<BFC> term_phase_row;  /* [ibox_r][local] */

    mutable std::vector<BFC> sigma_work_a;
    mutable std::vector<BFC> sigma_work_b;
};

static void precompute_factor(BFStrictFactor *F,
                              const float *tau,
                              const BFC *Amp)
{
    const int n = F->n;
    const int p = F->p;
    const int leaf_n = F->leaf_n;
    const int L = F->L;
    const double alpha = F->alpha;

    F->geometry = get_geometry_template(n, p, leaf_n);
    const BFGeometryTemplate &G = *F->geometry;

    const int nbox_s_leaf = 1 << L;
    F->init_phase_s.assign(
        (size_t)nbox_s_leaf * (size_t)leaf_n,
        c_make(0.f, 0.f));
    F->init_phase_node.assign(
        (size_t)nbox_s_leaf * (size_t)p,
        c_make(0.f, 0.f));

    const double r0_root = G.TR[(size_t)0][0].x0;
    for (int ibox_s = 0; ibox_s < nbox_s_leaf; ++ibox_s) {
        const BFBox1D &B = G.TS[(size_t)L][(size_t)ibox_s];
        for (int k = 0; k < leaf_n; ++k) {
            const int ps = B.i1 + k;
            const double t =
                interp_real_rowcol(tau, n, (double)ps, r0_root);
            F->init_phase_s[
                (size_t)ibox_s * (size_t)leaf_n + (size_t)k] =
                c_exp(alpha * t);
        }
        for (int it = 0; it < p; ++it) {
            const double t = interp_real_rowcol(
                tau, n, B.cheb[(size_t)it], r0_root);
            F->init_phase_node[
                (size_t)ibox_s * (size_t)p + (size_t)it] =
                c_exp(-alpha * t);
        }
    }

    F->lv_max = G.lv_max;
    F->r1.clear();
    F->r1.reserve(G.r1.size());
    for (const R1Geometry &GS : G.r1) {
        const int lv = GS.lv;
        R1Stage S;
        S.lv = GS.lv;
        S.nbox_r_now = GS.nbox_r_now;
        S.nbox_s_now = GS.nbox_s_now;
        S.nbox_r_prev = GS.nbox_r_prev;
        S.nbox_s_prev = GS.nbox_s_prev;
        S.L_child_to_parent = &GS.L_child_to_parent;
        S.phase_child.assign(
            (size_t)S.nbox_r_now * (size_t)S.nbox_s_prev *
                (size_t)p,
            c_make(0.f, 0.f));
        S.phase_parent.assign(
            (size_t)S.nbox_r_now * (size_t)S.nbox_s_now *
                (size_t)p,
            c_make(0.f, 0.f));

        for (int ibox_r = 0; ibox_r < S.nbox_r_now; ++ibox_r) {
            const BFBox1D &A = G.TR[(size_t)lv][(size_t)ibox_r];
            const double r0 = A.x0;
            for (int ibox_s_child = 0;
                 ibox_s_child < S.nbox_s_prev;
                 ++ibox_s_child) {
                const BFBox1D &Bc =
                    G.TS[(size_t)(L - lv + 1)]
                        [(size_t)ibox_s_child];
                for (int jt = 0; jt < p; ++jt) {
                    const double t = interp_real_rowcol(
                        tau, n, Bc.cheb[(size_t)jt], r0);
                    S.phase_child[
                        ((size_t)ibox_r * (size_t)S.nbox_s_prev +
                         (size_t)ibox_s_child) *
                            (size_t)p +
                        (size_t)jt] = c_exp(alpha * t);
                }
            }
            for (int ibox_s = 0;
                 ibox_s < S.nbox_s_now;
                 ++ibox_s) {
                const BFBox1D &B =
                    G.TS[(size_t)(L - lv)][(size_t)ibox_s];
                for (int it = 0; it < p; ++it) {
                    const double t = interp_real_rowcol(
                        tau, n, B.cheb[(size_t)it], r0);
                    S.phase_parent[
                        ((size_t)ibox_r * (size_t)S.nbox_s_now +
                         (size_t)ibox_s) *
                            (size_t)p +
                        (size_t)it] = c_exp(-alpha * t);
                }
            }
        }
        F->r1.push_back(std::move(S));
    }

    const int mid_lv = F->lv_max;
    F->nbox_r_mid = G.nbox_r_mid;
    F->nbox_s_mid = G.nbox_s_mid;
    F->switch_kernel.assign(
        (size_t)F->nbox_s_mid * (size_t)F->nbox_r_mid *
            (size_t)p * (size_t)p,
        c_make(0.f, 0.f));
    for (int ibox_s = 0; ibox_s < F->nbox_s_mid; ++ibox_s) {
        const BFBox1D &B =
            G.TS[(size_t)(L - mid_lv)][(size_t)ibox_s];
        for (int ibox_r = 0; ibox_r < F->nbox_r_mid; ++ibox_r) {
            const BFBox1D &A =
                G.TR[(size_t)mid_lv][(size_t)ibox_r];
            for (int it = 0; it < p; ++it) {
                const double rt = A.cheb[(size_t)it];
                for (int jt = 0; jt < p; ++jt) {
                    const double sj = B.cheb[(size_t)jt];
                    const double t =
                        interp_real_rowcol(tau, n, sj, rt);
                    const BFC amp =
                        interp_complex_rowcol(Amp, n, sj, rt);
                    F->switch_kernel[
                        (((size_t)ibox_s * (size_t)F->nbox_r_mid +
                          (size_t)ibox_r) *
                             (size_t)p +
                         (size_t)it) *
                            (size_t)p +
                        (size_t)jt] =
                        c_mul(amp, c_exp(alpha * t));
                }
            }
        }
    }

    F->r2.clear();
    F->r2.reserve(G.r2.size());
    for (const R2Geometry &GS : G.r2) {
        const int lv = GS.lv;
        R2Stage S;
        S.lv = GS.lv;
        S.nbox_r_now = GS.nbox_r_now;
        S.nbox_s_now = GS.nbox_s_now;
        S.nbox_r_prev = GS.nbox_r_prev;
        S.nbox_s_prev = GS.nbox_s_prev;
        S.L_target = &GS.L_target;
        S.phase_rp.assign(
            (size_t)S.nbox_r_now * (size_t)S.nbox_s_prev *
                (size_t)p,
            c_make(0.f, 0.f));
        S.phase_r.assign(
            (size_t)S.nbox_r_now * (size_t)S.nbox_s_prev *
                (size_t)p,
            c_make(0.f, 0.f));

        for (int ibox_r = 0; ibox_r < S.nbox_r_now; ++ibox_r) {
            const BFBox1D &A = G.TR[(size_t)lv][(size_t)ibox_r];
            const int ibox_r_parent = ibox_r / 2;
            const BFBox1D &Ap =
                G.TR[(size_t)(lv - 1)][(size_t)ibox_r_parent];
            for (int ibox_s_child = 0;
                 ibox_s_child < S.nbox_s_prev;
                 ++ibox_s_child) {
                const BFBox1D &Bc =
                    G.TS[(size_t)(L - lv + 1)]
                        [(size_t)ibox_s_child];
                const double s0 = Bc.x0;
                for (int jt = 0; jt < p; ++jt) {
                    const double t = interp_real_rowcol(
                        tau, n, s0, Ap.cheb[(size_t)jt]);
                    S.phase_rp[
                        ((size_t)ibox_r * (size_t)S.nbox_s_prev +
                         (size_t)ibox_s_child) *
                            (size_t)p +
                        (size_t)jt] = c_exp(-alpha * t);
                }
                for (int it = 0; it < p; ++it) {
                    const double t = interp_real_rowcol(
                        tau, n, s0, A.cheb[(size_t)it]);
                    S.phase_r[
                        ((size_t)ibox_r * (size_t)S.nbox_s_prev +
                         (size_t)ibox_s_child) *
                            (size_t)p +
                        (size_t)it] = c_exp(alpha * t);
                }
            }
        }
        F->r2.push_back(std::move(S));
    }

    const int nbox_r_leaf = 1 << L;
    F->term_phase_node.assign(
        (size_t)nbox_r_leaf * (size_t)p,
        c_make(0.f, 0.f));
    F->term_phase_row.assign(
        (size_t)nbox_r_leaf * (size_t)leaf_n,
        c_make(0.f, 0.f));
    const BFBox1D &Broot = G.TS[0][0];
    const double s0B = Broot.x0;
    for (int ibox_r = 0; ibox_r < nbox_r_leaf; ++ibox_r) {
        const BFBox1D &A = G.TR[(size_t)L][(size_t)ibox_r];
        for (int k = 0; k < leaf_n; ++k) {
            const int r = A.i1 + k;
            const double t =
                interp_real_rowcol(tau, n, s0B, (double)r);
            F->term_phase_row[
                (size_t)ibox_r * (size_t)leaf_n + (size_t)k] =
                c_exp(alpha * t);
        }
        for (int it = 0; it < p; ++it) {
            const double t = interp_real_rowcol(
                tau, n, s0B, A.cheb[(size_t)it]);
            F->term_phase_node[
                (size_t)ibox_r * (size_t)p + (size_t)it] =
                c_exp(-alpha * t);
        }
    }
}

static BFStrictFactor *create_strict_factor_from_contiguous(
    int n,
    const float *tau,
    const BFC *amp,
    float omega,
    int p,
    int leaf_n)
{
    if (n <= 0) bf_die("n must be positive.");
    if (p <= 0 || leaf_n <= 1) bf_die("invalid p or leaf_n.");
    if (p >= leaf_n) bf_die("need p < leaf_n.");
    if (n < leaf_n) bf_die("n must be >= leaf_n.");
    if (n % leaf_n != 0 || !is_power_of_two(n / leaf_n)) {
        bf_die("strict factor requires n = leaf_n * 2^L. Use segmented factor for arbitrary n.");
    }

    BFStrictFactor *F = new BFStrictFactor();
    F->n = n;
    F->p = p;
    F->leaf_n = leaf_n;
    F->L = int_log2_exact(n / leaf_n);
    F->alpha = -(double)omega;
    precompute_factor(F, tau, amp);
    return F;
}

BFStrictFactor *bf1d_strict_factor_create_phase_amp(int n,
                                                    float **tau_mat,
                                                    const fftwf_complex *Amp_mat,
                                                    float omega,
                                                    int p,
                                                    int leaf_n)
{
    if (n <= 0) bf_die("n must be positive.");
    if (p <= 0 || leaf_n <= 1) bf_die("invalid p or leaf_n.");
    if (p >= leaf_n) bf_die("need p < leaf_n.");
    if (n < leaf_n) bf_die("n must be >= leaf_n.");
    if (n % leaf_n != 0 || !is_power_of_two(n / leaf_n)) {
        bf_die("strict factor requires n = leaf_n * 2^L. Use segmented factor for arbitrary n.");
    }

    std::vector<float> tau((size_t)n * (size_t)n);
    std::vector<BFC> amp((size_t)n * (size_t)n);
    for (int r = 0; r < n; ++r) {
        for (int s = 0; s < n; ++s) {
            const size_t id = mat_index(r, s, n);
            tau[id] = tau_mat[r][s];
            amp[id] = c_make(Amp_mat[id][0], Amp_mat[id][1]);
        }
    }
    return create_strict_factor_from_contiguous(
        n, tau.data(), amp.data(), omega, p, leaf_n);
}

void bf1d_strict_factor_destroy(BFStrictFactor *F)
{
    delete F;
}

void bf1d_strict_factor_apply(const BFStrictFactor *F,
                              const fftwf_complex *Uin,
                              fftwf_complex *Uout)
{
    if (!F || !Uin || !Uout) return;
    int n = F->n;
    int p = F->p;
    int leaf_n = F->leaf_n;
    int L = F->L;
    memset(Uout, 0, (size_t)n * sizeof(fftwf_complex));

    size_t work_size = ((size_t)n / (size_t)leaf_n) * (size_t)p;
    if (F->sigma_work_a.size() != work_size) F->sigma_work_a.resize(work_size);
    if (F->sigma_work_b.size() != work_size) F->sigma_work_b.resize(work_size);

    std::vector<BFC> *sigma_prev = &F->sigma_work_a;
    std::vector<BFC> *sigma_curr = &F->sigma_work_b;
    const BFGeometryTemplate &G = *F->geometry;

    int nbox_s = 1 << L;

    for (int ibox_s = 0; ibox_s < nbox_s; ++ibox_s) {
        const BFBox1D &B = G.TS[(size_t)L][(size_t)ibox_s];
        for (int it = 0; it < p; ++it) {
            BFC acc = c_make(0.f, 0.f);
            for (int k = 0; k < leaf_n; ++k) {
                int ps = B.i1 + k;
                BFC u = c_make(Uin[ps][0], Uin[ps][1]);
                BFC ph = F->init_phase_s[(size_t)ibox_s * (size_t)leaf_n + (size_t)k];
                double Lval = G.init_L[((size_t)ibox_s * (size_t)leaf_n + (size_t)k) * (size_t)p + (size_t)it];
                c_acc(acc, c_scale(c_mul(u, ph), (float)Lval));
            }
            BFC phn = F->init_phase_node[(size_t)ibox_s * (size_t)p + (size_t)it];
            (*sigma_prev)[sig_idx(ibox_s, 0, it, 1, p)] = c_mul(acc, phn);
        }
    }

    for (size_t st = 0; st < F->r1.size(); ++st) {
        const R1Stage &S = F->r1[st];
        for (int ibox_r = 0; ibox_r < S.nbox_r_now; ++ibox_r) {
            int ibox_r_parent = ibox_r / 2;
            for (int ibox_s = 0; ibox_s < S.nbox_s_now; ++ibox_s) {
                BFC acc[128];
                if (p > 128) bf_die("p too large for stack buffer.");
                for (int it = 0; it < p; ++it) acc[it] = c_make(0.f,0.f);
                for (int child = 0; child < 2; ++child) {
                    int child_s = 2 * ibox_s + child;
                    for (int jt = 0; jt < p; ++jt) {
                        BFC oldv = (*sigma_prev)[sig_idx(child_s, ibox_r_parent, jt, S.nbox_r_prev, p)];
                        BFC ph = S.phase_child[((size_t)ibox_r * (size_t)S.nbox_s_prev + (size_t)child_s) * (size_t)p + (size_t)jt];
                        BFC tmp = c_mul(oldv, ph);
                        for (int it = 0; it < p; ++it) {
                            double Lval = (*S.L_child_to_parent)[(((size_t)ibox_s * 2u + (size_t)child) * (size_t)p + (size_t)jt) * (size_t)p + (size_t)it];
                            c_acc(acc[it], c_scale(tmp, (float)Lval));
                        }
                    }
                }
                for (int it = 0; it < p; ++it) {
                    BFC ph = S.phase_parent[((size_t)ibox_r * (size_t)S.nbox_s_now + (size_t)ibox_s) * (size_t)p + (size_t)it];
                    (*sigma_curr)[sig_idx(ibox_s, ibox_r, it, S.nbox_r_now, p)] = c_mul(acc[it], ph);
                }
            }
        }
        std::swap(sigma_prev, sigma_curr);
    }

    for (int ibox_s = 0; ibox_s < F->nbox_s_mid; ++ibox_s) {
        for (int ibox_r = 0; ibox_r < F->nbox_r_mid; ++ibox_r) {
            for (int it = 0; it < p; ++it) {
                BFC sum = c_make(0.f, 0.f);
                for (int jt = 0; jt < p; ++jt) {
                    BFC k = F->switch_kernel[(((size_t)ibox_s * (size_t)F->nbox_r_mid + (size_t)ibox_r) * (size_t)p + (size_t)it) * (size_t)p + (size_t)jt];
                    BFC b = (*sigma_prev)[sig_idx(ibox_s, ibox_r, jt, F->nbox_r_mid, p)];
                    c_acc(sum, c_mul(k, b));
                }
                (*sigma_curr)[sig_idx(ibox_s, ibox_r, it, F->nbox_r_mid, p)] = sum;
            }
        }
    }
    std::swap(sigma_prev, sigma_curr);

    for (size_t st = 0; st < F->r2.size(); ++st) {
        const R2Stage &S = F->r2[st];
        for (int ibox_r = 0; ibox_r < S.nbox_r_now; ++ibox_r) {
            int ibox_r_parent = ibox_r / 2;
            for (int ibox_s = 0; ibox_s < S.nbox_s_now; ++ibox_s) {
                BFC acc[128];
                if (p > 128) bf_die("p too large for stack buffer.");
                for (int it = 0; it < p; ++it) acc[it] = c_make(0.f,0.f);
                for (int child = 0; child < 2; ++child) {
                    int child_s = 2 * ibox_s + child;
                    BFC tmp[128], y[128];
                    for (int jt = 0; jt < p; ++jt) {
                        BFC oldv = (*sigma_prev)[sig_idx(child_s, ibox_r_parent, jt, S.nbox_r_prev, p)];
                        BFC ph = S.phase_rp[((size_t)ibox_r * (size_t)S.nbox_s_prev + (size_t)child_s) * (size_t)p + (size_t)jt];
                        tmp[jt] = c_mul(oldv, ph);
                    }
                    for (int it = 0; it < p; ++it) {
                        y[it] = c_make(0.f,0.f);
                        for (int jt = 0; jt < p; ++jt) {
                            double Lval = (*S.L_target)[((size_t)ibox_r * (size_t)p + (size_t)it) * (size_t)p + (size_t)jt];
                            c_acc(y[it], c_scale(tmp[jt], (float)Lval));
                        }
                        BFC ph = S.phase_r[((size_t)ibox_r * (size_t)S.nbox_s_prev + (size_t)child_s) * (size_t)p + (size_t)it];
                        c_acc(acc[it], c_mul(y[it], ph));
                    }
                }
                for (int it = 0; it < p; ++it) {
                    (*sigma_curr)[sig_idx(ibox_s, ibox_r, it, S.nbox_r_now, p)] = acc[it];
                }
            }
        }
        std::swap(sigma_prev, sigma_curr);
    }

    int nbox_r_leaf = 1 << L;
    for (int ibox_r = 0; ibox_r < nbox_r_leaf; ++ibox_r) {
        const BFBox1D &A = G.TR[(size_t)L][(size_t)ibox_r];
        BFC tmp[128];
        if (p > 128) bf_die("p too large for stack buffer.");
        for (int it = 0; it < p; ++it) {
            BFC oldv = (*sigma_prev)[sig_idx(0, ibox_r, it, nbox_r_leaf, p)];
            BFC ph = F->term_phase_node[(size_t)ibox_r * (size_t)p + (size_t)it];
            tmp[it] = c_mul(oldv, ph);
        }
        for (int k = 0; k < leaf_n; ++k) {
            BFC sum = c_make(0.f,0.f);
            for (int it = 0; it < p; ++it) {
                double Lval = G.term_L[((size_t)ibox_r * (size_t)leaf_n + (size_t)k) * (size_t)p + (size_t)it];
                c_acc(sum, c_scale(tmp[it], (float)Lval));
            }
            BFC ph = F->term_phase_row[(size_t)ibox_r * (size_t)leaf_n + (size_t)k];
            BFC out = c_mul(sum, ph);
            int r = A.i1 + k;
            Uout[r][0] = out.r;
            Uout[r][1] = out.i;
        }
    }
}

struct BFPanel {
    int row0;
    int col0;
    int rows;
    int cols;
    int nloc;

    /*
     * If exact_leaf is set, this panel is evaluated by the exact leaf-stage
     * kernel stored below.  This is not an error-based fallback; it is a
     * deterministic leaf evaluation used by the segmented butterfly factor.
     * It is essential for matrices with hard aperture/anti-alias masks, where
     * forcing Chebyshev interpolation across the zero/nonzero boundary causes
     * large high-frequency errors.
     */
    int exact_leaf;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<BFC> kval;

    BFStrictFactor *factor;
    mutable std::vector<BFC> uin_loc;
    mutable std::vector<BFC> uout_loc;
};

struct BFStrictSegmentedFactor {
    int n;
    int p;
    int leaf_n;
    int panel_n;
    std::vector<BFPanel> panels;
};

static void zero_bfc_vec(std::vector<BFC> &v)
{
    for (size_t i = 0; i < v.size(); ++i) {
        v[i].r = 0.0f;
        v[i].i = 0.0f;
    }
}


struct ActivePrefix {
    int matrix_n;
    const fftwf_complex *amplitude;
    float eps2;
    int row0;
    int col0;
    int rows;
    int cols;
    std::vector<int> prefix;

    ActivePrefix(int matrix_n,
                 const fftwf_complex *Amp_mat,
                 float amp_eps,
                 int maximum_rows,
                 int maximum_cols)
        : matrix_n(matrix_n),
          amplitude(Amp_mat),
          eps2(amp_eps * amp_eps),
          row0(0), col0(0), rows(0), cols(0)
    {
        prefix.reserve((size_t)(maximum_rows + 1) *
                       (size_t)(maximum_cols + 1));
    }

    void reset(int first_row, int first_col, int row_count, int col_count)
    {
        row0 = first_row;
        col0 = first_col;
        rows = row_count;
        cols = col_count;
        prefix.assign((size_t)(rows + 1) * (size_t)(cols + 1), 0);
        const int stride = cols + 1;
        for (int r = 0; r < rows; ++r) {
            int row_sum = 0;
            for (int c = 0; c < cols; ++c) {
                const size_t id = mat_index(row0 + r, col0 + c,
                                            matrix_n);
                const float ar = amplitude[id][0];
                const float ai = amplitude[id][1];
                row_sum += ar * ar + ai * ai > eps2 ? 1 : 0;
                prefix[(size_t)(r + 1) * (size_t)stride +
                       (size_t)(c + 1)] =
                    prefix[(size_t)r * (size_t)stride +
                           (size_t)(c + 1)] +
                    row_sum;
            }
        }
    }

    int count(int row0, int col0, int rows, int cols) const
    {
        const int stride = this->cols + 1;
        const int local_row0 = row0 - this->row0;
        const int local_col0 = col0 - this->col0;
        const int row1 = local_row0 + rows;
        const int col1 = local_col0 + cols;
        return prefix[(size_t)row1 * (size_t)stride +
                      (size_t)col1] -
               prefix[(size_t)local_row0 * (size_t)stride +
                      (size_t)col1] -
               prefix[(size_t)row1 * (size_t)stride +
                      (size_t)local_col0] +
               prefix[(size_t)local_row0 * (size_t)stride +
                      (size_t)local_col0];
    }
};

static void make_exact_panel(BFStrictSegmentedFactor *SF,
                             int n,
                             float **tau_mat,
                             const fftwf_complex *Amp_mat,
                             float omega,
                             int row0,
                             int col0,
                             int rows,
                             int cols,
                             float amp_eps)
{
    BFPanel P;
    P.row0 = row0;
    P.col0 = col0;
    P.rows = rows;
    P.cols = cols;
    P.nloc = std::max(rows, cols);
    P.exact_leaf = 1;
    P.factor = nullptr;
    P.row_ptr.assign((size_t)rows + 1, 0);
    P.col_idx.reserve((size_t)rows * (size_t)cols);
    P.kval.reserve((size_t)rows * (size_t)cols);

    float eps2 = amp_eps * amp_eps;
    for (int r = 0; r < rows; ++r) {
        P.row_ptr[(size_t)r] = (int)P.col_idx.size();
        for (int c = 0; c < cols; ++c) {
            size_t idg = mat_index(row0 + r, col0 + c, n);
            float ar = Amp_mat[idg][0];
            float ai = Amp_mat[idg][1];
            if (ar * ar + ai * ai <= eps2) continue;

            const float phase = omega * tau_mat[row0 + r][col0 + c];
            /* K = Amp * exp(-i*phase).  c_exp() uses one sincos evaluation
             * instead of two independent transcendental calls. */
            const BFC k = c_mul(c_make(ar, ai), c_exp(-(double)phase));

            P.col_idx.push_back(c);
            P.kval.push_back(k);
        }
    }
    P.row_ptr[(size_t)rows] = (int)P.col_idx.size();
    SF->panels.push_back(std::move(P));
}

static void make_strict_bf_panel(BFStrictSegmentedFactor *SF,
                                 int n,
                                 float **tau_mat,
                                 const fftwf_complex *Amp_mat,
                                 float omega,
                                 int p,
                                 int leaf_n,
                                 int row0,
                                 int col0,
                                 int rows,
                                 int cols)
{
    const int nloc = rows; /* called only for square legal panels */
    std::vector<float> tau_storage((size_t)nloc * (size_t)nloc);
    std::vector<BFC> amp_storage((size_t)nloc * (size_t)nloc);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t idg = mat_index(row0 + r, col0 + c, n);
            const size_t idl = mat_index(r, c, nloc);
            tau_storage[idl] = tau_mat[row0 + r][col0 + c];
            amp_storage[idl] = c_make(Amp_mat[idg][0], Amp_mat[idg][1]);
        }
    }

    BFPanel P;
    P.row0 = row0;
    P.col0 = col0;
    P.rows = rows;
    P.cols = cols;
    P.nloc = nloc;
    P.exact_leaf = 0;
    /* The local panel is already contiguous here.  Calling the public
     * row-pointer API would copy tau and amplitude a second time. */
    P.factor = create_strict_factor_from_contiguous(
        nloc, tau_storage.data(), amp_storage.data(), omega, p, leaf_n);
    P.uin_loc.resize((size_t)nloc);
    P.uout_loc.resize((size_t)nloc);
    zero_bfc_vec(P.uin_loc);
    zero_bfc_vec(P.uout_loc);

    SF->panels.push_back(std::move(P));
}


static double block_residual_phase_estimate(int /*n*/,
                                            float **tau_mat,
                                            int row0,
                                            int col0,
                                            int rows,
                                            int cols,
                                            float omega)
{
    if (rows <= 1 || cols <= 1) return 0.0;

    /* Estimate the nonseparable phase residual in this panel.  A pure
     * row-plus-column phase has zero residual and is ideal for butterfly
     * compression.  Large residuals require further panel subdivision.
     */
    int r1 = row0;
    int r2 = row0 + rows - 1;
    int c1 = col0;
    int c2 = col0 + cols - 1;

    double t00 = (double)tau_mat[r1][c1];
    double max_res = 0.0;

    for (int ir = 0; ir < 5; ++ir) {
        int rr;
        if (ir == 0) rr = r1;
        else if (ir == 1) rr = r2;
        else if (ir == 2) rr = row0 + rows / 2;
        else if (ir == 3) rr = row0 + rows / 4;
        else rr = row0 + (3 * rows) / 4;
        if (rr < r1) rr = r1;
        if (rr > r2) rr = r2;

        for (int jc = 0; jc < 5; ++jc) {
            int cc;
            if (jc == 0) cc = c1;
            else if (jc == 1) cc = c2;
            else if (jc == 2) cc = col0 + cols / 2;
            else if (jc == 3) cc = col0 + cols / 4;
            else cc = col0 + (3 * cols) / 4;
            if (cc < c1) cc = c1;
            if (cc > c2) cc = c2;

            double approx = (double)tau_mat[rr][c1] + (double)tau_mat[r1][cc] - t00;
            double res = std::fabs((double)omega * ((double)tau_mat[rr][cc] - approx));
            if (res > max_res) max_res = res;
        }
    }
    return max_res;
}

static bool is_legal_strict_panel(int rows, int cols, int leaf_n)
{
    if (rows != cols) return false;
    if (rows < 2 * leaf_n) return false; /* leaf panels are handled exactly. */
    if (rows % leaf_n != 0) return false;
    return is_power_of_two(rows / leaf_n);
}

static void add_adaptive_panel(BFStrictSegmentedFactor *SF,
                               int n,
                               float **tau_mat,
                               const fftwf_complex *Amp_mat,
                               const ActivePrefix &active,
                               float omega,
                               int p,
                               int leaf_n,
                               int row0,
                               int col0,
                               int rows,
                               int cols,
                               float amp_eps,
                               float phase_tol)
{
    if (rows <= 0 || cols <= 0) return;

    const int nnz = active.count(row0, col0, rows, cols);
    if (nnz == 0) return;
    const bool full = nnz == rows * cols;

    /* Fully active, square, legal panels use the true multi-level Chebyshev butterfly.
     * This is where p controls the approximation accuracy and cost.
     */
    if (full && is_legal_strict_panel(rows, cols, leaf_n) &&
        block_residual_phase_estimate(n, tau_mat, row0, col0, rows, cols, omega) <= (double)phase_tol) {
        make_strict_bf_panel(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                             row0, col0, rows, cols);
        return;
    }

    /* Mixed panels contain a hard aperture/anti-alias zero boundary.  They are
     * recursively split until leaf size, where interactions are evaluated exactly.
     * This avoids Chebyshev interpolation across discontinuities while still using
     * the true butterfly recursion on smooth fully active panels.
     */
    if (rows <= leaf_n && cols <= leaf_n) {
        make_exact_panel(SF, n, tau_mat, Amp_mat, omega, row0, col0, rows, cols, amp_eps);
        return;
    }

    if (rows >= cols && rows > leaf_n) {
        int r1 = rows / 2;
        int r2 = rows - r1;
        add_adaptive_panel(SF, n, tau_mat, Amp_mat, active, omega, p, leaf_n,
                           row0, col0, r1, cols, amp_eps, phase_tol);
        add_adaptive_panel(SF, n, tau_mat, Amp_mat, active, omega, p, leaf_n,
                           row0 + r1, col0, r2, cols, amp_eps, phase_tol);
    } else if (cols > leaf_n) {
        int c1 = cols / 2;
        int c2 = cols - c1;
        add_adaptive_panel(SF, n, tau_mat, Amp_mat, active, omega, p, leaf_n,
                           row0, col0, rows, c1, amp_eps, phase_tol);
        add_adaptive_panel(SF, n, tau_mat, Amp_mat, active, omega, p, leaf_n,
                           row0, col0 + c1, rows, c2, amp_eps, phase_tol);
    } else {
        make_exact_panel(SF, n, tau_mat, Amp_mat, omega, row0, col0, rows, cols, amp_eps);
    }
}

static void add_adaptive_panel_dense(BFStrictSegmentedFactor *SF,
                                     int n,
                                     float **tau_mat,
                                     const fftwf_complex *Amp_mat,
                                     float omega,
                                     int p,
                                     int leaf_n,
                                     int row0,
                                     int col0,
                                     int rows,
                                     int cols,
                                     float amp_eps,
                                     float phase_tol)
{
    if (rows <= 0 || cols <= 0) return;

    if (is_legal_strict_panel(rows, cols, leaf_n) &&
        block_residual_phase_estimate(
            n, tau_mat, row0, col0, rows, cols, omega) <=
            (double)phase_tol) {
        make_strict_bf_panel(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                             row0, col0, rows, cols);
        return;
    }

    if (rows <= leaf_n && cols <= leaf_n) {
        make_exact_panel(SF, n, tau_mat, Amp_mat, omega,
                         row0, col0, rows, cols, amp_eps);
        return;
    }

    if (rows >= cols && rows > leaf_n) {
        const int r1 = rows / 2;
        const int r2 = rows - r1;
        add_adaptive_panel_dense(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                                 row0, col0, r1, cols, amp_eps, phase_tol);
        add_adaptive_panel_dense(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                                 row0 + r1, col0, r2, cols,
                                 amp_eps, phase_tol);
    } else if (cols > leaf_n) {
        const int c1 = cols / 2;
        const int c2 = cols - c1;
        add_adaptive_panel_dense(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                                 row0, col0, rows, c1, amp_eps, phase_tol);
        add_adaptive_panel_dense(SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                                 row0, col0 + c1, rows, c2,
                                 amp_eps, phase_tol);
    } else {
        make_exact_panel(SF, n, tau_mat, Amp_mat, omega,
                         row0, col0, rows, cols, amp_eps);
    }
}

static BFStrictSegmentedFactor *create_segmented_phase_amp_impl(
    int n,
    float **tau_mat,
    const fftwf_complex *Amp_mat,
    float omega,
    int p,
    int leaf_n,
    int panel_levels,
    float amp_eps,
    float phase_tol,
    bool known_fully_active)
{
    if (n <= 0) bf_die("n must be positive.");
    if (p <= 0 || leaf_n <= 1) bf_die("invalid p or leaf_n.");
    if (p >= leaf_n) bf_die("need p < leaf_n.");
    if (amp_eps < 0.0f) amp_eps = 0.0f;
    if (phase_tol <= 0.0f || !std::isfinite(phase_tol)) phase_tol = (float)BF_PHASE_RES_TOL;

    BFStrictSegmentedFactor *SF = new BFStrictSegmentedFactor();
    SF->n = n;
    SF->p = p;
    SF->leaf_n = leaf_n;
    SF->panel_n = panel_size_from_levels(leaf_n, panel_levels);
    if (SF->panel_n < leaf_n) SF->panel_n = leaf_n;
    if (SF->panel_n > n) SF->panel_n = next_legal_size(n, leaf_n);

    int panel_n = SF->panel_n;
    /* Avoid repeated growth of the panel vector.  The adaptive tree can have
     * at most one stored panel per leaf_n-by-leaf_n matrix tile. */
    const size_t leaf_rows =
        ((size_t)n + (size_t)leaf_n - 1u) / (size_t)leaf_n;
    SF->panels.reserve(leaf_rows * leaf_rows);
    /* Build the summed-area activity map one top-level panel at a time.  The
     * previous n-by-n prefix occupied 4 MiB for n=1024 in every concurrent
     * builder and streamed another complete matrix through memory.  Adaptive
     * recursion never crosses a top-level panel, so a panel-local prefix has
     * identical decisions while normally fitting in L1 cache. */
    if (known_fully_active) {
        /* The caller filled this amplitude matrix and verified every sample is
         * above amp_eps.  Skip the second O(n^2) activity-prefix pass; the
         * adaptive phase decisions are otherwise identical. */
        for (int row0 = 0; row0 < n; row0 += panel_n) {
            const int rows = std::min(panel_n, n - row0);
            for (int col0 = 0; col0 < n; col0 += panel_n) {
                const int cols = std::min(panel_n, n - col0);
                add_adaptive_panel_dense(
                    SF, n, tau_mat, Amp_mat, omega, p, leaf_n,
                    row0, col0, rows, cols, amp_eps, phase_tol);
            }
        }
    } else {
        ActivePrefix active(n, Amp_mat, amp_eps, panel_n, panel_n);
        for (int row0 = 0; row0 < n; row0 += panel_n) {
            const int rows = std::min(panel_n, n - row0);
            for (int col0 = 0; col0 < n; col0 += panel_n) {
                const int cols = std::min(panel_n, n - col0);
                active.reset(row0, col0, rows, cols);
                add_adaptive_panel(
                    SF, n, tau_mat, Amp_mat, active, omega, p, leaf_n,
                    row0, col0, rows, cols, amp_eps, phase_tol);
            }
        }
    }

    return SF;
}

BFStrictSegmentedFactor *bf1d_strict_segmented_create_phase_amp(
    int n,
    float **tau_mat,
    const fftwf_complex *Amp_mat,
    float omega,
    int p,
    int leaf_n,
    int panel_levels,
    float amp_eps,
    float phase_tol)
{
    return create_segmented_phase_amp_impl(
        n, tau_mat, Amp_mat, omega, p, leaf_n, panel_levels,
        amp_eps, phase_tol, false);
}

BFStrictSegmentedFactor *bf1d_strict_segmented_create_phase_amp_dense(
    int n,
    float **tau_mat,
    const fftwf_complex *Amp_mat,
    float omega,
    int p,
    int leaf_n,
    int panel_levels,
    float amp_eps,
    float phase_tol)
{
    return create_segmented_phase_amp_impl(
        n, tau_mat, Amp_mat, omega, p, leaf_n, panel_levels,
        amp_eps, phase_tol, true);
}

void bf1d_strict_segmented_destroy(BFStrictSegmentedFactor *F)
{
    if (!F) return;
    for (size_t i = 0; i < F->panels.size(); ++i) {
        if (F->panels[i].factor) {
            bf1d_strict_factor_destroy(F->panels[i].factor);
        }
    }
    delete F;
}

void bf1d_strict_segmented_apply(const BFStrictSegmentedFactor *F,
                                 const fftwf_complex *Uin,
                                 fftwf_complex *Uout)
{
    if (!F || !Uin || !Uout) return;
    memset(Uout, 0, (size_t)F->n * sizeof(fftwf_complex));
    for (size_t ip = 0; ip < F->panels.size(); ++ip) {
        const BFPanel &P = F->panels[ip];

        if (P.exact_leaf) {
            for (int r = 0; r < P.rows; ++r) {
                BFC sum = c_make(0.0f, 0.0f);
                int b = P.row_ptr[(size_t)r];
                int e = P.row_ptr[(size_t)r + 1];
                for (int q = b; q < e; ++q) {
                    int c = P.col_idx[(size_t)q];
                    BFC k = P.kval[(size_t)q];
                    BFC u = c_make(Uin[P.col0 + c][0], Uin[P.col0 + c][1]);
                    c_acc(sum, c_mul(k, u));
                }
                Uout[P.row0 + r][0] += sum.r;
                Uout[P.row0 + r][1] += sum.i;
            }
            continue;
        }

        for (int c = 0; c < P.cols; ++c) {
            P.uin_loc[(size_t)c].r = Uin[P.col0 + c][0];
            P.uin_loc[(size_t)c].i = Uin[P.col0 + c][1];
        }

        bf1d_strict_factor_apply(P.factor,
                                 reinterpret_cast<const fftwf_complex *>(P.uin_loc.data()),
                                 reinterpret_cast<fftwf_complex *>(P.uout_loc.data()));

        for (int r = 0; r < P.rows; ++r) {
            Uout[P.row0 + r][0] += P.uout_loc[(size_t)r].r;
            Uout[P.row0 + r][1] += P.uout_loc[(size_t)r].i;
        }
    }
}

void bf1d_strict_segmented_apply_pair(const BFStrictSegmentedFactor *F,
                                      const fftwf_complex *Uin_a,
                                      const fftwf_complex *Uin_b,
                                      fftwf_complex *Uout_a,
                                      fftwf_complex *Uout_b)
{
    if (!F || !Uin_a || !Uin_b || !Uout_a || !Uout_b) return;
    memset(Uout_a, 0, (size_t)F->n * sizeof(fftwf_complex));
    memset(Uout_b, 0, (size_t)F->n * sizeof(fftwf_complex));
    for (size_t ip = 0; ip < F->panels.size(); ++ip) {
        const BFPanel &P = F->panels[ip];
        if (!P.exact_leaf) {
            /* The strict recursion owns mutable work buffers.  Keep its
             * proven implementation, while still fusing the numerous exact
             * aperture panels below. */
            for (int c = 0; c < P.cols; ++c) {
                P.uin_loc[(size_t)c] = c_make(Uin_a[P.col0 + c][0],
                                              Uin_a[P.col0 + c][1]);
            }
            bf1d_strict_factor_apply(P.factor,
                reinterpret_cast<const fftwf_complex *>(P.uin_loc.data()),
                reinterpret_cast<fftwf_complex *>(P.uout_loc.data()));
            for (int r = 0; r < P.rows; ++r) {
                Uout_a[P.row0 + r][0] += P.uout_loc[(size_t)r].r;
                Uout_a[P.row0 + r][1] += P.uout_loc[(size_t)r].i;
            }
            for (int c = 0; c < P.cols; ++c) {
                P.uin_loc[(size_t)c] = c_make(Uin_b[P.col0 + c][0],
                                              Uin_b[P.col0 + c][1]);
            }
            bf1d_strict_factor_apply(P.factor,
                reinterpret_cast<const fftwf_complex *>(P.uin_loc.data()),
                reinterpret_cast<fftwf_complex *>(P.uout_loc.data()));
            for (int r = 0; r < P.rows; ++r) {
                Uout_b[P.row0 + r][0] += P.uout_loc[(size_t)r].r;
                Uout_b[P.row0 + r][1] += P.uout_loc[(size_t)r].i;
            }
            continue;
        }

        for (int r = 0; r < P.rows; ++r) {
            BFC sum_a = c_make(0.0f, 0.0f);
            BFC sum_b = c_make(0.0f, 0.0f);
            const int begin = P.row_ptr[(size_t)r];
            const int end = P.row_ptr[(size_t)r + 1];
            for (int q = begin; q < end; ++q) {
                const int column = P.col0 + P.col_idx[(size_t)q];
                const BFC kernel = P.kval[(size_t)q];
                c_acc(sum_a, c_mul(kernel,
                    c_make(Uin_a[column][0], Uin_a[column][1])));
                c_acc(sum_b, c_mul(kernel,
                    c_make(Uin_b[column][0], Uin_b[column][1])));
            }
            Uout_a[P.row0 + r][0] += sum_a.r;
            Uout_a[P.row0 + r][1] += sum_a.i;
            Uout_b[P.row0 + r][0] += sum_b.r;
            Uout_b[P.row0 + r][1] += sum_b.i;
        }
    }
}

void butterfly_apply_1d_phase_amp(int n,
                                  float **tau_mat,
                                  const fftwf_complex *Amp_mat,
                                  const fftwf_complex *Uin,
                                  fftwf_complex *Uout,
                                  float omega,
                                  int p,
                                  int leaf_n)
{
    /*
     * Safe one-shot strict butterfly entry.  For speed in the main program,
     * prefer bf1d_strict_segmented_create_phase_amp() once and then call
     * bf1d_strict_segmented_apply() repeatedly.
     */
    BFStrictSegmentedFactor *F = bf1d_strict_segmented_create_phase_amp(
        n, tau_mat, Amp_mat, omega, p, leaf_n, 1, 0.0f, (float)BF_PHASE_RES_TOL);
    bf1d_strict_segmented_apply(F, Uin, Uout);
    bf1d_strict_segmented_destroy(F);
}

#ifdef BF1D_STRICT_SELF_TEST
static void direct_apply(int n, float **tau, const fftwf_complex *Amp, const fftwf_complex *U, fftwf_complex *O, float omega)
{
    for (int r = 0; r < n; ++r) { O[r][0] = 0.f; O[r][1] = 0.f; }
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            size_t id = mat_index(r,c,n);
            float ar = Amp[id][0], ai = Amp[id][1];
            float ph = omega * tau[r][c];
            float cp = std::cos(ph), sp = std::sin(ph);
            float kr = ar * cp + ai * sp;
            float ki = ai * cp - ar * sp;
            O[r][0] += kr * U[c][0] - ki * U[c][1];
            O[r][1] += kr * U[c][1] + ki * U[c][0];
        }
    }
}
int main()
{
    int n = 64, leaf = 16, p = 12;
    float omega = 20.0f;
    std::vector<float> tau_storage((size_t)n*n);
    std::vector<float*> tau((size_t)n);
    fftwf_complex *Amp = (fftwf_complex *)std::calloc((size_t)n*n, sizeof(fftwf_complex));
    fftwf_complex *U = (fftwf_complex *)std::calloc((size_t)n, sizeof(fftwf_complex));
    fftwf_complex *Od = (fftwf_complex *)std::calloc((size_t)n, sizeof(fftwf_complex));
    fftwf_complex *Ob = (fftwf_complex *)std::calloc((size_t)n, sizeof(fftwf_complex));
    for (int r = 0; r < n; ++r) tau[(size_t)r] = &tau_storage[(size_t)r*n];
    for (int r = 0; r < n; ++r) {
        U[(size_t)r][0] = std::sin(0.1f*r); U[(size_t)r][1] = std::cos(0.2f*r);
        for (int c = 0; c < n; ++c) {
            tau[(size_t)r][c] = 0.5f + 0.01f*std::sqrt(1.0f + (float)((r-c)*(r-c)));
            size_t id = mat_index(r,c,n);
            float a = std::exp(-0.001f*(float)((r-c)*(r-c)));
            Amp[id][0] = a; Amp[id][1] = 0.1f*a;
        }
    }
    direct_apply(n, tau.data(), Amp, U, Od, omega);
    BFStrictSegmentedFactor *F = bf1d_strict_segmented_create_phase_amp(n, tau.data(), Amp, omega, p, leaf, 2, 0.0f);
    bf1d_strict_segmented_apply(F, U, Ob);
    double num=0, den=0;
    for (int i = 0; i < n; ++i) {
        double dr = Ob[i][0] - Od[i][0];
        double di = Ob[i][1] - Od[i][1];
        num += dr*dr + di*di;
        den += (double)Od[i][0]*Od[i][0] + (double)Od[i][1]*Od[i][1];
    }
    std::printf("relerr=%g panels=%zu\n", std::sqrt(num/den), F->panels.size());
    bf1d_strict_segmented_destroy(F);
    std::free(Amp); std::free(U); std::free(Od); std::free(Ob);
    return 0;
}
#endif
