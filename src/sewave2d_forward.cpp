#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include "../SEWAVE/sewave2d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

const char* laplace_name(int type_laplace)
{
    return type_laplace == 1 ? "pseudospectral" : "FD8";
}

void check_stability(const sewave::Grid2D& velocity,
                     float dt,
                     int type_laplace)
{
    if (velocity.v.empty() || dt <= 0.0f ||
        velocity.dx <= 0.0f || velocity.dz <= 0.0f) {
        return;
    }

    const float vmax =
        *std::max_element(velocity.v.begin(), velocity.v.end());
    double limit = 0.0;
    if (type_laplace == 1) {
        const double kmax = std::sqrt(
            std::pow(M_PI / velocity.dx, 2.0) +
            std::pow(M_PI / velocity.dz, 2.0));
        limit = 2.0 / (vmax * kmax);
    } else {
        const double inverse_h = std::sqrt(
            1.0 / (velocity.dx * velocity.dx) +
            1.0 / (velocity.dz * velocity.dz));
        limit = 0.45 / (vmax * inverse_h);
    }

    INFO(("forward stability: method=%s vmax=%g dt=%g estimated_limit=%g",
          laplace_name(type_laplace), vmax, dt, limit));
    if (dt > limit) {
        ERROR(("unstable forward-modeling time step: dt=%g exceeds estimated limit=%g",
               dt, limit));
    }
}

} // namespace

int main(int argc, char** argv)
{
    se_par_init(argc, argv);
    try {
        if (!se_have_par("velocity")) {
            ERROR(("Need velocity= RSF model"));
        }
        if (!se_have_par("output")) {
            ERROR(("Need output= RSF shot gather"));
        }

        const char* velocity_path = se_get_par_str("velocity");
        const char* output_path = se_get_par_str("output");
        sewave::Grid2D velocity = sewave::read_rsf2d(velocity_path);

        sewave::ForwardParams p;
        p.nt = se_have_par("nt") ? se_get_par_int("nt") : 6000;
        p.dt = se_have_par("dt") ? se_get_par_float("dt") : 0.0007f;
        p.fdom = se_have_par("fdom") ? se_get_par_float("fdom") : 20.0f;
        p.sx = se_have_par("sx")
            ? se_get_par_float("sx")
            : velocity.x0 + 0.5f * (velocity.nx - 1) * velocity.dx;
        p.sz = se_have_par("sz") ? se_get_par_float("sz") : velocity.z0;
        p.rz = se_have_par("rz") ? se_get_par_float("rz") : velocity.z0;
        p.nr = se_have_par("nr") ? se_get_par_int("nr") : velocity.nx;
        p.r0 = se_have_par("r0") ? se_get_par_float("r0") : velocity.x0;
        p.dr = se_have_par("dr") ? se_get_par_float("dr") : velocity.dx;
        p.ns = se_have_par("ns") ? se_get_par_int("ns") : 1;
        p.ds = se_have_par("ds") ? se_get_par_float("ds") : 0.0f;
        p.nbc = se_have_par("nbc") ? se_get_par_int("nbc") : 40;
        p.L = se_have_par("L") ? se_get_par_int("L") : 30;
        p.alpha = se_have_par("alpha") ? se_get_par_float("alpha") : 1.0f;
        p.type_compute_laplace = se_have_par("type_compute_Laplace")
            ? se_get_par_int("type_compute_Laplace")
            : 0;
        p.visco = false;
        p.type = 0;
        p.flag_smooth = se_have_par("flag_smooth")
            ? se_get_par_int("flag_smooth") != 0
            : false;
        p.flag_homo = se_have_par("flag_homo")
            ? se_get_par_int("flag_homo") != 0
            : false;
        p.progress_interval = se_have_par("progress_interval")
            ? se_get_par_int("progress_interval")
            : std::max(1, p.nt / 10);

#ifdef _OPENMP
        INFO(("OpenMP max threads=%d", omp_get_max_threads()));
#endif
        INFO(("forward model=%s output=%s", velocity_path, output_path));
        INFO(("geometry sx=%g sz=%g rz=%g nr=%d r0=%g dr=%g",
              p.sx, p.sz, p.rz, p.nr, p.r0, p.dr));
        INFO(("sampling nt=%d dt=%g fdom=%g homogeneous=%d",
              p.nt, p.dt, p.fdom, p.flag_homo ? 1 : 0));

        check_stability(velocity, p.dt, p.type_compute_laplace);
        sewave::Data3D data = sewave::forward(velocity, p, nullptr);
        sewave::write_rsf3d(output_path, data);
        INFO(("forward modeling completed"));
        return 0;
    } catch (const std::exception& error) {
        ERROR(("sewave2d_forward failed: %s", error.what()));
    }
    return 1;
}
