#include <SERECKIRCH/include/huygens_cli.hpp>
#include "program_help.hpp"

#include <SERECKIRCH/include/huygens_sweep.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    if (kirch_help::show_if_requested(argc, argv)) return 0;

    try {
        huygens_cli::initialize(argc, argv);
        const std::string velocity = huygens_cli::required_string("velocity");
        const std::string output = huygens_cli::optional_string(
            "output", "huygens_blocks.txt");
        const int first_block_rows = huygens_cli::optional_int("first_block_rows", 3);
        const int block_rows = huygens_cli::optional_int("block_rows", 64);
        const int overlap_rows = huygens_cli::optional_int("overlap_rows", 0);

        const se::huygens::Model2D model = se::huygens::read_velocity_model(velocity);
        const se::huygens::BlockInfo info = se::huygens::make_block_info(
            model, first_block_rows, block_rows, overlap_rows);
        se::huygens::write_block_info(output, info);

        std::cout << "Generated " << info.blocks.size() << " blocks\n"
                  << "Block file: " << output << '\n'
                  << "Propagation overlap rows: " << info.overlap_rows << '\n'
                  << "Official target intervals are contiguous; source_iz is moved "
                     "upward into the preceding completed interval.\n";
        for (const se::huygens::Block& block : info.blocks) {
            std::cout << "block=" << block.id
                      << " start=" << block.start_iz
                      << " source=" << block.source_iz
                      << " target=[" << block.target_start_iz
                      << ',' << block.target_end_iz << ']'
                      << " halo_end=" << block.halo_end_iz << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "huygens_block_info: " << error.what() << '\n';
        return 1;
    }
}
