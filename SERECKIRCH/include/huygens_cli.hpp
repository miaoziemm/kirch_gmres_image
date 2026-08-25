#ifndef HUYGENS_CLI_HPP
#define HUYGENS_CLI_HPP

#include <SEBASIC/include/se_basic.h>
#include <SEFILESYSTEM/include/se_fs.h>


#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace huygens_cli {

inline std::string required_string(const char* name)
{
    if (!se_have_par(name)) {
        throw std::invalid_argument(std::string("missing required parameter: ") + name + "=");
    }
    return std::string(se_get_par_str(name));
}

inline std::string optional_string(const char* name, const std::string& value)
{
    return se_have_par(name) ? std::string(se_get_par_str(name)) : value;
}

inline int optional_int(const char* name, int value)
{
    return se_have_par(name) ? se_get_par_int(name) : value;
}

inline float optional_float(const char* name, float value)
{
    return se_have_par(name) ? se_get_par_float(name) : value;
}

inline void initialize(int argc, char** argv)
{
    se_par_init(argc, argv);
}

inline void set_openmp_threads(int threads)
{
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

} // namespace huygens_cli

#endif
