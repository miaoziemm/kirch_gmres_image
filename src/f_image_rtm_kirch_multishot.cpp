#include "program_help.hpp"
#include <SERECKIRCH/include/global_bf_imaging_workflow.hpp>

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;
    return kirch::imaging::with_global_bf_imaging_workflow(
        argc, argv,
        [](kirch::imaging::GlobalBfImagingWorkflow& imaging) {
            imaging.calculate_receiver_wavefields();
#pragma omp parallel for schedule(dynamic) \
    num_threads(imaging.frequency_parallelism())
            for (std::size_t frequency = 0;
                 frequency < imaging.frequency_count(); ++frequency) {
                imaging.process_frequency(frequency);
            }
            imaging.finish();
        });
}
