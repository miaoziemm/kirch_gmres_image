#ifndef KIRCH_GLOBAL_BF_IMAGING_WORKFLOW_HPP
#define KIRCH_GLOBAL_BF_IMAGING_WORKFLOW_HPP

#include <cstddef>
#include <functional>
#include <memory>

namespace kirch::imaging {

class GlobalBfImagingWorkflow {
public:
    GlobalBfImagingWorkflow(GlobalBfImagingWorkflow&&) noexcept;
    GlobalBfImagingWorkflow& operator=(GlobalBfImagingWorkflow&&) noexcept;
    ~GlobalBfImagingWorkflow();

    std::size_t shot_count() const;
    std::size_t frequency_count() const;

    void calculate_receiver_wavefields();
    void begin_frequency(std::size_t frequency);
    void calculate_source_wavefield(std::size_t shot, std::size_t frequency);
    void iteratively_correct_wavefields(std::size_t shot, std::size_t frequency);
    void cross_correlate_image(std::size_t shot, std::size_t frequency);
    void end_frequency(std::size_t frequency);
    void finish();

private:
    struct Impl;
    explicit GlobalBfImagingWorkflow(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend int with_global_bf_imaging_workflow(
        int, char**, const std::function<void(GlobalBfImagingWorkflow&)>&);
};

int with_global_bf_imaging_workflow(
    int argc, char** argv,
    const std::function<void(GlobalBfImagingWorkflow&)>& orchestration);

} // namespace kirch::imaging

#endif
