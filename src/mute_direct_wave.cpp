#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include "../SEWAVE/sewave2d.h"
#include "program_help.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;
    se_par_init(argc, argv);
    try {
        if (!se_have_par("input") || !se_have_par("output"))
            ERROR(("Need input= and output= RSF shot gathers"));
        const float velocity = se_have_par("direct_velocity")
            ? se_get_par_float("direct_velocity") : 1.5f;
        const float source_depth = se_have_par("source_depth")
            ? se_get_par_float("source_depth") : 0.0f;
        const float receiver_depth = se_have_par("receiver_depth")
            ? se_get_par_float("receiver_depth") : 0.0f;
        const float extra_time = se_have_par("extra_time")
            ? se_get_par_float("extra_time") : 0.08f;
        const float taper_time = se_have_par("taper_time")
            ? se_get_par_float("taper_time") : 0.04f;
        if (!(velocity > 0.0f) || extra_time < 0.0f || taper_time < 0.0f)
            ERROR(("invalid direct-wave mute parameters"));

        sewave::Data3D data = sewave::read_rsf3d(se_get_par_str("input"));
        constexpr float pi = 3.14159265358979323846f;
        for (int shot = 0; shot < data.ns; ++shot) {
            const float sx = data.s0 + shot * data.ds;
            for (int receiver = 0; receiver < data.nr; ++receiver) {
                const float rx = data.r0 + receiver * data.dr;
                const float distance = std::hypot(
                    rx - sx, receiver_depth - source_depth);
                const float mute_end = distance / velocity + extra_time;
                for (int it = 0; it < data.nt; ++it) {
                    const float time = data.t0 + it * data.dt;
                    float weight = 1.0f;
                    if (time <= mute_end) weight = 0.0f;
                    else if (taper_time > 0.0f && time < mute_end + taper_time) {
                        const float q = (time - mute_end) / taper_time;
                        weight = 0.5f - 0.5f * std::cos(pi * q);
                    }
                    const std::size_t index =
                        (static_cast<std::size_t>(shot) * data.nr + receiver) *
                        data.nt + it;
                    data.d[index] *= weight;
                }
            }
        }
        sewave::write_rsf3d(se_get_par_str("output"), data);
        INFO(("direct-wave mute completed: velocity=%g extra=%g taper=%g",
              velocity, extra_time, taper_time));
        return 0;
    } catch (const std::exception& error) {
        ERROR(("mute_direct_wave failed: %s", error.what()));
    }
    return 1;
}
