#include "program_help.hpp"
#include <SERECKIRCH/include/global_bf_imaging_workflow.hpp>

#include <cstddef>

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;
    return kirch::imaging::with_global_bf_imaging_workflow(
        argc, argv,
        [](kirch::imaging::GlobalBfImagingWorkflow& imaging) {
            imaging.calculate_receiver_wavefields();
            for (std::size_t frequency = 0;
                 frequency < imaging.frequency_count(); ++frequency) {
                imaging.begin_frequency(frequency);
                for (std::size_t shot = 0; shot < imaging.shot_count(); ++shot) {
                    imaging.calculate_source_wavefield(shot, frequency);
                    imaging.iteratively_correct_wavefields(shot, frequency);
                    imaging.cross_correlate_image(shot, frequency);
                }
                imaging.end_frequency(frequency);
            }
            imaging.finish();
        });
}
