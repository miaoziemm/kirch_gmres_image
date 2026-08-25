#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_par_sep.h>
#include "../SEWAVE/sewave2d.h"
#include "program_help.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;
    se_par_init(argc, argv);
    try {
        if (!se_have_par("input") || !se_have_par("output"))
            ERROR(("Need input= and output= RSF velocity models"));
        const float sigma = se_have_par("sigma") ? se_get_par_float("sigma") : 8.0f;
        if (!(sigma > 0.0f)) ERROR(("sigma must be positive"));

        sewave::Grid2D model = sewave::read_rsf2d(se_get_par_str("input"));
        const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
        std::vector<float> kernel(static_cast<std::size_t>(2 * radius + 1));
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            const float q = static_cast<float>(k) / sigma;
            kernel[static_cast<std::size_t>(k + radius)] = std::exp(-0.5f * q * q);
            sum += kernel[static_cast<std::size_t>(k + radius)];
        }
        for (float& value : kernel) value /= sum;

        std::vector<float> temp(model.v.size());
        std::vector<float> smooth(model.v.size());
        auto index = [&model](int iz, int ix) {
            return static_cast<std::size_t>(ix) * model.nz + iz;
        };
        for (int ix = 0; ix < model.nx; ++ix)
            for (int iz = 0; iz < model.nz; ++iz) {
                float value = 0.0f;
                for (int k = -radius; k <= radius; ++k)
                    value += kernel[static_cast<std::size_t>(k + radius)] *
                        model.v[index(std::clamp(iz + k, 0, model.nz - 1), ix)];
                temp[index(iz, ix)] = value;
            }
        for (int ix = 0; ix < model.nx; ++ix)
            for (int iz = 0; iz < model.nz; ++iz) {
                float value = 0.0f;
                for (int k = -radius; k <= radius; ++k)
                    value += kernel[static_cast<std::size_t>(k + radius)] *
                        temp[index(iz, std::clamp(ix + k, 0, model.nx - 1))];
                smooth[index(iz, ix)] = value;
            }

        sewave::write_rsf2d(se_get_par_str("output"), model, smooth, "Smoothed velocity");
        INFO(("smoothed velocity model written with sigma=%g grid samples", sigma));
        return 0;
    } catch (const std::exception& error) {
        ERROR(("smooth_velocity_model failed: %s", error.what()));
    }
    return 1;
}
