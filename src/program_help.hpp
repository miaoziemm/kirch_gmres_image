#ifndef KIRCH_PROGRAM_HELP_HPP
#define KIRCH_PROGRAM_HELP_HPP

#include <iostream>
#include <string>

namespace kirch_help {

inline std::string executable_name(const char* path)
{
    const std::string value = path != nullptr ? path : "";
    const std::size_t slash = value.find_last_of("/\\");
    return slash == std::string::npos ? value : value.substr(slash + 1);
}

inline bool requested(int argc, char** argv)
{
    if (argc <= 1) return true;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i] != nullptr ? argv[i] : "";
        if (argument == "-h" || argument == "--help" ||
            argument == "help" || argument == "help=1" ||
            argument == "help=true" || argument == "help=yes") {
            return true;
        }
    }
    return false;
}

inline const char* common_footer()
{
    return R"KHELP(
通用帮助入口
------------
  不带任何参数运行程序
  -h
  --help
  help=1

参数格式统一为 key=value。RSF 文件通常由一个文本头文件和其 in= 指向的
float 二进制文件组成。模型和地震数据坐标单位必须保持一致。
)KHELP";
}

inline const char* program_text(const std::string& program)
{
    if (program == "huygens_block_info") {
        return R"KHELP(
huygens_block_info
==================
功能
----
根据二维 RSF 速度模型建立递归 Kirchhoff/Huygens 深度分块文件。正式输出
深度区间连续；overlap_rows 通过把下一块的传播基准面上移到已完成区域中
实现重叠，不会重复写入成像/波场网格。

用法
----
  huygens_block_info velocity=MODEL.rsf [key=value ...]

必选输入
--------
  velocity=FILE
      二维速度模型，RSF 轴顺序 n1=z、n2=x。
      来源：外部速度建模或模型生成程序。
      下游：同一 velocity 文件还要传给全部走时、波场和成像程序。

可选参数
--------
  output=FILE                 默认 huygens_blocks.txt
      输出分块描述文件。
  first_block_rows=N          默认 3
      第一块正式包含的深度网格行数，至少为 3。
  block_rows=N                默认 64
      后续每块新增的正式输出行数，至少为 2。
  overlap_rows=N              默认 0
      下一传播基准面向上一已完成块内上移的行数；
      必须小于 block_rows，且不大于 first_block_rows-2。

输出及程序连接
--------------
  output
      下游参数 block_file=FILE，供以下程序读取：
      travel_time_solver
      travel_time_solver_oneway
      wave_frequency_direct
      wave_frequency_bf
      wave_frequency_direct_oneway
      wave_frequency_bf_oneway
      frequency_kirchhoff_imaging_direct
      frequency_kirchhoff_imaging_bf
)KHELP";
    }

    if (program == "travel_time_solver") {
        return R"KHELP(
travel_time_solver
==================
功能
----
为完整 Cauchy-Kirchhoff 分块传播预计算三类表：走时 tau、几何振幅 amp
和传播基准面法向走时导数 dtaun。每个基准面源执行中心、上方和下方三次
FMM，用于构造法向导数。

用法
----
  travel_time_solver velocity=MODEL.rsf block_file=BLOCKS.txt [key=value ...]

必选输入
--------
  velocity=FILE
      二维 RSF 速度模型；通常与 huygens_block_info 的 velocity 相同。
  block_file=FILE
      分块描述；来源为 huygens_block_info 的 output。

可选参数
--------
  output_prefix=PREFIX        默认 huygens_tt/travel
      三类输出表的共同前缀。
  timing=FILE
      默认 output_prefix + "_timing.rsf"。
  source_stride=N             默认 1
      传播基准面横向源点抽样步长。
  target_x_stride=N           默认 1
      目标横向抽样步长。
  target_z_stride=N           默认 1
      目标深度抽样步长。
  include_propagation_overlap=0|1  默认 0
      全局成像启用 Scheme-A 拼接时设置为 1，使走时表覆盖传播重叠行。
  threads=N                   默认 0
      OpenMP 线程数；0 使用运行环境默认值。
  max_table_mb=MB             默认 8192
      单块 tau+amp+dtaun 内存保护上限。

输出及程序连接
--------------
  PREFIX_block_NNN_tau.rsf
  PREFIX_block_NNN_amp.rsf
  PREFIX_block_NNN_dtaun.rsf
      将同一个 PREFIX 作为下游 table_prefix=PREFIX，供：
      wave_frequency_direct
      wave_frequency_bf
  timing
      FMM 时间、表内存、源点数和目标点数诊断；不作为其他程序必选输入。

注意
----
该程序生成完整 Cauchy 积分所需的三类表。当前全局成像流程只读取其中
的 tau 表；第零块使用解析相移初始化，不生成也不读取第零块走时表。
)KHELP";
    }

    if (program == "travel_time_solver_oneway") {
        return R"KHELP(
travel_time_solver_oneway
=========================
功能
----
为不需要格林函数梯度的单向频率域 Kirchhoff 积分预计算走时表。每个
基准面源只执行一次 FMM。

用法
----
  travel_time_solver_oneway velocity=MODEL.rsf block_file=BLOCKS.txt [key=value ...]

必选输入
--------
  velocity=FILE
      二维 RSF 速度模型。
  block_file=FILE
      来源为 huygens_block_info 的 output。

可选参数
--------
  output_prefix=PREFIX        默认 huygens_tt/travel
      输出走时表共同前缀。
  timing=FILE
      默认 output_prefix + "_timing.rsf"。
  source_stride=N             默认 1
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
      基准面源、目标 x 和目标 z 的抽样步长。
  threads=N                   默认 0
      OpenMP 线程数；0 使用运行环境默认值。
  include_first_block=0|1     默认 0
      0：第一块由解析震源场初始化，只生成后续块表；
      1：同时生成地表到第一块内部的表，频率域成像必须使用 1。
  max_table_mb=MB             默认 8192
      单块走时表内存保护上限。

输出及程序连接
--------------
  PREFIX_block_NNN_tau.rsf
      将 PREFIX 作为 table_prefix=PREFIX：
      include_first_block=0 时供 wave_frequency_direct_oneway 和
      wave_frequency_bf_oneway 使用；
      include_first_block=1 时供 frequency_kirchhoff_imaging_direct 和
      frequency_kirchhoff_imaging_bf 使用。
  timing
      FMM 时间、走时表内存和采样规模诊断。
)KHELP";
    }

    if (program == "wave_frequency_direct") {
        return R"KHELP(
wave_frequency_direct
=====================
功能
----
使用完整 Cauchy-Kirchhoff 边界积分直接求和，递归计算一个频率的二维
复波场。第一块采用解析 Hankel 点源场，后续块使用 U 和 dU/dn 两个边界
通道。

用法
----
  wave_frequency_direct velocity=MODEL.rsf block_file=BLOCKS.txt \
      table_prefix=PREFIX [key=value ...]

必选输入
--------
  velocity=FILE
      二维速度模型。
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver 的 output_prefix；必须存在 tau、amp、
      dtaun 三类表。

可选参数
--------
  output_real=FILE            默认 wave_frequency_direct_real.rsf
  output_imag=FILE            默认 wave_frequency_direct_imag.rsf
  timing=FILE                 默认 wave_frequency_direct_timing.rsf
  frequency=HZ                默认 25
  source_amplitude=A          默认 1
  source_radius=R             默认 0；0 表示 0.5*min(dx,dz)
  source_ix=N                 默认 -1；-1 表示模型横向中心
  source_iz=N                 默认 0；当前原型要求为 0
  source_stride=N             默认 1；必须与走时表一致
  target_x_stride=N           默认 1；当前递归输出要求为 1
  target_z_stride=N           默认 1；当前递归输出要求为 1
  threads=N                   默认 0；使用 OpenMP 默认线程数

输出及程序连接
--------------
  output_real/output_imag
      同一复波场的实部和虚部。可与 wave_frequency_bf 的对应输出比较；
      model/compare_and_plot_wavefields.py 可用于绘图和误差检查。
  timing
      每块直接积分时间和矩阵规模诊断。
)KHELP";
    }

    if (program == "wave_frequency_bf") {
        return R"KHELP(
wave_frequency_bf
=================
功能
----
使用 ButterflyPACK 压缩完整 Cauchy-Kirchhoff 两个边界通道，递归计算
一个频率的二维复波场，并输出构树、压缩、apply、内存和秩统计。

用法
----
  wave_frequency_bf velocity=MODEL.rsf block_file=BLOCKS.txt \
      table_prefix=PREFIX [key=value ...]

必选输入
--------
  velocity=FILE
      二维速度模型。
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver 的 output_prefix，包含 tau/amp/dtaun。

可选参数
--------
  output_real=FILE            默认 wave_frequency_bf_real.rsf
  output_imag=FILE            默认 wave_frequency_bf_imag.rsf
  timing=FILE                 默认 wave_frequency_bf_timing.rsf
  frequency=HZ                默认 25
  source_amplitude=A          默认 1
  source_radius=R             默认 0；0 表示 0.5*min(dx,dz)
  source_ix=N                 默认 -1；-1 表示 nx/2
  source_iz=N                 默认 0
  source_stride=N             默认 1；必须与走时表一致
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
  tol=EPS                     默认 1e-4；ButterflyPACK 压缩容差
  leaf=N                      默认 64；层次树叶节点大小，至少为 4
  verbosity=N                 默认 0；小于 0 隐藏 ButterflyPACK 原始输出
  threads=N                   默认 0；OpenMP 默认线程数

输出及程序连接
--------------
  output_real/output_imag
      BF 复波场实部/虚部；与 wave_frequency_direct 对应输出比较。
  timing
      构树、压缩、apply、压缩内存、峰值内存、最大秩和采样元素数。
)KHELP";
    }

    if (program == "wave_frequency_direct_oneway") {
        return R"KHELP(
wave_frequency_direct_oneway
============================
功能
----
使用仅依赖走时和边界波场值的单向 Kirchhoff 积分直接求和。支持在同一
速度模型、频率和分块下计算多个 RHS/震源。

用法
----
  wave_frequency_direct_oneway velocity=MODEL.rsf block_file=BLOCKS.txt \
      table_prefix=PREFIX [key=value ...]

必选输入
--------
  velocity=FILE
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver_oneway 的 output_prefix；
      本程序通常使用 include_first_block=0 生成的表。

可选参数
--------
  output_real=FILE            默认 wave_frequency_direct_real.rsf
  output_imag=FILE            默认 wave_frequency_direct_imag.rsf
  timing=FILE                 默认 wave_frequency_direct_timing.rsf
  frequency=HZ                默认 25
  filter_dt=SEC               默认 0.001
  filter_length=SEC           默认 0.025
  filter_lookup_subsamples=N  默认 64
  source_amplitude=A          默认 1
  source_radius=R             默认 0；0 表示 0.5*min(dx,dz)
  source_ix=N                 默认 -1；rhs_count=1 时表示 nx/2
  source_iz=N                 默认 0
  rhs_count=N                 默认 1
      大于 1 时在整个横向范围均匀设置多个震源。
  output_rhs=N                默认 -1；-1 表示 rhs_count/2
      指定写出的 RHS，编号从 0 开始。
  source_stride=N             默认 1
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
  threads=N                   默认 0

输出及程序连接
--------------
  output_real/output_imag
      output_rhs 对应复波场，可与 wave_frequency_bf_oneway 比较。
  timing
      每块直接积分时间、源点/目标点数和 RHS 数。
)KHELP";
    }

    if (program == "wave_frequency_bf_oneway") {
        return R"KHELP(
wave_frequency_bf_oneway
========================
功能
----
将单向 Kirchhoff 核写成缓变振幅乘 exp(-i*omega*tau)，使用相位感知
分段 bf1d 蝶形因子计算一个频率的多个 RHS。每个目标深度的因子构建一次，
随后对全部 RHS 复用。

用法
----
  wave_frequency_bf_oneway velocity=MODEL.rsf block_file=BLOCKS.txt \
      table_prefix=PREFIX [key=value ...]

必选输入
--------
  velocity=FILE
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver_oneway 的 output_prefix；
      本程序通常使用 include_first_block=0。

公共参数
--------
  output_real=FILE            默认 wave_frequency_bf_real.rsf
  output_imag=FILE            默认 wave_frequency_bf_imag.rsf
  timing=FILE                 默认 wave_frequency_bf_timing.rsf
  frequency=HZ                默认 25
  filter_dt=SEC               默认 0.001
  filter_length=SEC           默认 0.025
  filter_lookup_subsamples=N  默认 64
  source_amplitude=A          默认 1
  source_radius=R             默认 0
  source_ix=N                 默认 -1
  source_iz=N                 默认 0
  rhs_count=N                 默认 1
  output_rhs=N                默认 -1；-1 表示 rhs_count/2
  source_stride=N             默认 1；当前 bf1d 路径要求为 1
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
  threads=N                   默认 0

蝶形参数
--------
  bf_p=N                      默认 12；Chebyshev 插值阶数
  bf_n_leaf=N                 默认 16；叶节点大小，且 bf_p < bf_n_leaf
  bf_panel_levels=N           默认 1；分段面板层数
  bf_amp_eps=EPS              默认 1e-20；振幅截断阈值
  bf_phase_tol=EPS            默认 1.0；相位残差容差

输出及程序连接
--------------
  output_real/output_imag
      output_rhs 的 BF 复波场；与 wave_frequency_direct_oneway 比较。
  timing
      每块因子构建、全部 RHS apply、单 RHS apply 和规模统计。
)KHELP";
    }

    if (program == "frequency_kirchhoff_imaging_direct") {
        return R"KHELP(
frequency_kirchhoff_imaging_direct
==================================
功能
----
读取时间域炮集，逐道 FFT；生成并 FFT Ricker 震源；对选定正频率逐块
直接 Kirchhoff 外推震源场和共轭检波场，并累加
Re{Us * conj(Ur)} 得到频率域互相关成像结果。

用法
----
  frequency_kirchhoff_imaging_direct velocity=MODEL.rsf \
      block_file=BLOCKS.txt table_prefix=PREFIX \
      seismic_data=DATA.rsf [key=value ...]

必选输入及来源
--------------
  velocity=FILE
      二维速度模型。
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver_oneway 的 output_prefix；
      必须以 include_first_block=1 生成。
  seismic_data=FILE
      时间域 RSF 地震数据。轴 1=time，轴 2=offset/receiver，
      可选轴 3=shot。来源为正演程序或实测预处理数据。

数据几何
--------
  cmp=0|1                     默认 1
      1：axis2 为偏移距，receiver_x=shot_x+axis2；
      0：axis2 为绝对检波点坐标，receiver_x=axis2。
  shot_begin=N                默认 0
  shot_count=N                默认 -1；-1 表示从 shot_begin 到末炮
  shot_stride=N               默认 1
  aperture_trace=N            默认 -1；负值不限制道数
  aperture_distance=D         默认 -1；负值不限制物理距离

震源和频率参数
--------------
  fdom=HZ                     默认 20；Ricker 主频
  source_time=SEC             默认 -1；负值取 data.t0+1.5/fdom
  source_amplitude=A          默认 1
  nfft=N                      默认 0；0 取不小于 nt 的最小 2 次幂
  fmin=HZ                     默认 3
  fmax=HZ                     默认 45，并自动受 Nyquist 限制
  frequency_stride=N          默认 1；正频率 FFT bin 抽样步长
  filter_dt=SEC               默认 0；0 使用地震数据 dt
  filter_length=SEC           默认 0.025
  filter_lookup_subsamples=N  默认 64

空间、内存和成像参数
--------------------
  source_stride=N             默认 1；当前成像要求为 1
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
  threads=N                   默认 0
  max_state_mb=MB             默认 8192；全部炮边界频谱状态内存保护
  source_normalization=0|1    默认 0
  normalization_epsilon=EPS   默认 1e-6；照明归一化稳定项

输出及程序连接
--------------
  image=FILE
      默认 frequency_kirchhoff_imaging_direct_image.rsf，最终二维成像结果。
      可与 frequency_kirchhoff_imaging_bf 的 image 比较。
  illumination=FILE
      默认空字符串，不输出；设置后写出震源照明。
  timing=FILE
      默认 frequency_kirchhoff_imaging_direct_timing.rsf；
      记录 FFT、滤波器准备、传播/成像时间和跳过的检波点数。
)KHELP";
    }

    if (program == "frequency_kirchhoff_imaging_bf") {
        return R"KHELP(
frequency_kirchhoff_imaging_bf
==============================
功能
----
执行与 frequency_kirchhoff_imaging_direct 相同的频率域互相关成像，但
采用“频率小批次 × 深度小批次”的低内存相位感知 bf1d 因子库。每个
小批次在所有炮之间复用，完成 apply 后立即释放，因此峰值内存不再随
全部频率数和整块深度数的乘积增长。

用法
----
  frequency_kirchhoff_imaging_bf velocity=MODEL.rsf \
      block_file=BLOCKS.txt table_prefix=PREFIX \
      seismic_data=DATA.rsf [key=value ...]

必选输入及上游
--------------
  velocity=FILE
  block_file=FILE
      来源为 huygens_block_info 的 output。
  table_prefix=PREFIX
      来源为 travel_time_solver_oneway，且 include_first_block=1。
  seismic_data=FILE
      时间域 RSF，轴 1=time、轴 2=offset/receiver、轴 3=shot 可选。

公共参数及默认值
----------------
  cmp=1
  shot_begin=0
  shot_count=-1
  shot_stride=1
  aperture_trace=-1
  aperture_distance=-1
  fdom=20
  source_time=-1
  source_amplitude=1
  nfft=0
  fmin=3
  fmax=45
  frequency_stride=1
  filter_dt=0
  filter_length=0.025
  filter_lookup_subsamples=64
  source_stride=1
  target_x_stride=1
  target_z_stride=1
  threads=0
  max_state_mb=8192
  source_normalization=0
  normalization_epsilon=1e-6
      含义与 frequency_kirchhoff_imaging_direct 完全相同。
      cmp=1 表示 shot+offset；cmp=0 表示绝对 receiver 坐标。

蝶形因子库参数
--------------
  bf_p=N                      默认 12
  bf_n_leaf=N                 默认 16
  bf_panel_levels=N           默认 1
  bf_amp_eps=EPS              默认 1e-20
  bf_phase_tol=EPS            默认 0.5
  bf_build_threads=N          默认 0；0 根据 OpenMP 和工作区自动选择
  bf_build_workspace_mb=MB    默认 512；并行构建临时工作区上限
  bf_frequency_batch=N        默认 4；同时驻留的频率批次大小
  bf_depth_batch=N            默认 1；同时驻留的目标深度批次大小
  bf_max_factor_count=N       默认 200000；驻留批次因子数保护上限
      峰值驻留因子数约为 bf_frequency_batch × bf_depth_batch。
      减小两个 batch 参数可以直接降低因子和走时临时内存。

输出及程序连接
--------------
  image=FILE
      默认 frequency_kirchhoff_imaging_bf_image.rsf；
      与 frequency_kirchhoff_imaging_direct 的 image 做精度对比。
  illumination=FILE
      默认空，不写；设置后输出震源照明。
  timing=FILE
      默认 frequency_kirchhoff_imaging_bf_timing.rsf；
      分离 FFT、滤波器、BF 构建、BF apply 和成像时间。
)KHELP";
    }

    if (program == "kirchhoff_butterfly_rsf") {
        return R"KHELP(
kirchhoff_butterfly_rsf
=======================
功能
----
独立的 1D 基准面到 2D 波场实验程序。在同一次运行中计算完整
Cauchy-Kirchhoff 直接结果和 ButterflyPACK 结果，并输出走时、误差、
时间和内存统计。它不依赖 huygens_block_info。

用法
----
  kirchhoff_butterfly_rsf velocity=MODEL.rsf [key=value ...]

必选输入
--------
  velocity=FILE
      二维 RSF 速度模型，n1=z、n2=x；来源为外部模型。

几何和震源参数
--------------
  frequency=HZ                默认 25
  source_ix=N                 默认 -1；-1 表示 nx/2
  source_iz=N                 默认 0
  datum_iz=N                  默认 1；Kirchhoff 基准面深度行
  top_layers=N                默认 3；至少为 3
  boundary_stride=N           默认 16
  target_x_stride=N           默认 1
  target_z_stride=N           默认 1
  source_radius=R             默认 0；0 表示 0.5*min(dx,dz)
  source_amplitude=A          默认 1

ButterflyPACK 和内存参数
-----------------------
  tol=EPS                     默认 1e-4
  leaf=N                      默认 64
  verbosity=N                 默认 0
  max_traveltime_mb=MB        默认 16384
  threads=N                   默认 0

输出及默认值
------------
  direct_real=FILE
      默认 kirchhoff_direct_25hz_real.rsf
  direct_imag=FILE
      默认 kirchhoff_direct_25hz_imag.rsf
  butterfly_real=FILE
      默认 kirchhoff_butterfly_25hz_real.rsf
  butterfly_imag=FILE
      默认 kirchhoff_butterfly_25hz_imag.rsf
      四个波场文件用于直接法/BF 绘图和误差对比；当前无其他程序把它们
      作为必选输入。
  traveltime=FILE
      默认 kirchhoff_traveltime.rsf
  traveltime_derivative=FILE
      默认 kirchhoff_traveltime_normal_derivative.rsf
      两者是本次独立实验的诊断输出，不等同于分块程序的 table_prefix。
  timing=FILE
      默认 kirchhoff_25hz_timing.txt；记录时间、误差、压缩内存和秩。
)KHELP";
    }

    if (program == "smooth_velocity_model") {
        return R"KHELP(
smooth_velocity_model
=====================
Gaussian-smooth a 2-D RSF velocity model for traveltime calculation and imaging.

Usage:
  smooth_velocity_model input=MODEL.rsf output=SMOOTH.rsf [sigma=8]
)KHELP";
    }

    if (program == "mute_direct_wave") {
        return R"KHELP(
mute_direct_wave
================
Apply a geometry-based front mute to direct arrivals in an RSF shot gather.

Usage:
  mute_direct_wave input=DATA.rsf output=MUTED.rsf [key=value ...]

Options:
  direct_velocity=1.5 source_depth=0 receiver_depth=0
  extra_time=0.08 taper_time=0.04
)KHELP";
    }

    if (program == "sewave2d_forward") {
        return R"KHELP(
sewave2d_forward
================
功能
----
使用二维声学有限差分或伪谱算子进行时间域正演，并将多炮检波记录写成
RSF 数据。程序会检查时间步长稳定性。

用法
----
  sewave2d_forward velocity=MODEL.rsf output=DATA.rsf [key=value ...]

必选参数
--------
  velocity=FILE                 二维 RSF 速度模型，n1=z、n2=x
  output=FILE                   输出炮集，n1=time、n2=receiver、n3=shot

主要可选参数
------------
  nt=N                          默认 6000
  dt=SEC                        默认 0.0007
  fdom=HZ                       默认 20
  sx=X  sz=Z                    首炮震源坐标
  ns=N  ds=D                    炮数和炮间距，默认 1、0
  rz=Z  nr=N  r0=X  dr=D        检波线几何
  nbc=N                         吸收边界宽度，默认 40
  L=N                           空间算子长度，默认 30
  alpha=A                       吸收边界强度，默认 1
  type_compute_Laplace=0|1      0=FD8，1=伪谱；默认 0
  flag_smooth=0|1               是否平滑速度模型，默认 0
  flag_homo=0|1                 是否使用均匀模型，默认 0
  progress_interval=N           进度输出间隔，默认 nt/10

输出连接
--------
  output 可直接作为 frequency_kirchhoff_imaging_bf_gmres 的
  seismic_data=FILE 输入。
)KHELP";
    }

    if (program == "f_image_rtm_kirch") {
        return R"KHELP(
f_image_rtm_kirch
=================
功能
----
复刻参考点源频率域 Kirchhoff RTM 流程：递归 ButterflyPACK 检波波场、
点源射线初场、全局 Helmholtz GMRES 矫正以及频率域互相关叠加成像。

用法
----
  f_image_rtm_kirch velocity=MODEL.rsf block_file=BLOCKS.txt \
      table_prefix=PREFIX seismic_data=DATA.rsf [key=value ...]

关键参数
--------
  cmp=0|1 shot_begin=0 shot_count=-1 shot_stride=1
  fdom=20 source_time=-1 source_amplitude=1
  nfft=0 fmin=3 fmax=45 frequency_stride=1
  global_source_z=Z global_source_correction_z0=Z
  gmres_restart=30 gmres_outer=10
      使用参考程序完全相同的无预条件 restarted GMRES、固定 25 点
      吸收边界、1e-12 停止阈值和高阶 Helmholtz 空间离散。
  bpack_tol=1e-4 bpack_leaf=64
  image=FILE illumination=FILE timing=FILE
  image_before_correction=FILE
  illumination_before_correction=FILE
  source_wavefield_output_dir=DIR
      启用后，逐炮逐频率输出 source_ray、source_gmres、source_update、
      receiver_kirchhoff、receiver_gmres 和 receiver_update 的实部与虚部
      RSF；其中 ray/kirchhoff 是矫正前结果，gmres 是矫正后结果。
)KHELP";
    }

    if (program == "frequency_kirchhoff_imaging_bf_gmres" ||
        program == "frequency_kirchhoff_imaging_bf_global_gmres") {
        return R"KHELP(
frequency_kirchhoff_imaging_bf_gmres
====================================
功能
----
仅保留生产全局矫正流程：由 eFMM 走时构造 A*exp(-i*omega*T) 震源初场，
使用递归矩形 ButterflyPACK 计算检波波场，在完整模型上分别进行移位
Laplacian + SuperLU ILUTP 预条件 GMRES 全局矫正，最后互相关成像。

用法
----
  frequency_kirchhoff_imaging_bf_gmres velocity=MODEL.rsf \
      block_file=BLOCKS.txt table_prefix=PREFIX \
      seismic_data=DATA.rsf [key=value ...]

必选参数
--------
  velocity=FILE                 二维 RSF 速度模型
  block_file=FILE               huygens_block_info 的 output
  table_prefix=PREFIX           逐块单向走时表前缀
  seismic_data=FILE             正演或实测时间域炮集

数据、频率与输出
----------------
  cmp=0|1                       默认 1；1=炮点+偏移距
  shot_begin=0 shot_count=-1 shot_stride=1
  fdom=20 source_time=-1 source_amplitude=1
  nfft=0 fmin=3 fmax=45 frequency_stride=1
  image=FILE                    输出最终图像
  illumination=FILE             可选震源照明输出
  timing=FILE                   分块构建、传播和成像计时
  threads=N                     默认 0，使用 OpenMP 默认值

ButterflyPACK 检波波场参数
--------------------------
  bpack_tol=1e-4 bpack_leaf=64 bpack_depth_chunk=20
  bpack_lrlevel=100 bpack_sample_para=1.5
  bpack_pat_comp=1 bpack_less_adapt=1 bpack_rdetect_factor=0.3
  bpack_reuse_tree=1 bpack_geometry_cache_mb=1024

全局矫正参数
------------
  gmres_enable=0|1              默认 1
  global_gmres_iterations=10
  gmres_restart=30 gmres_tolerance=1e-8
  gmres_nabs=25 gmres_damp_max=2
      吸收层在原始模型四周向外扩展，迭代完成后裁回原始模型大小。
  global_smooth_sigma=1 global_shift_beta=0.30
  global_ilut_drop_tolerance=0.03
  global_ilut_fill_factor=12
  global_ilut_pivot_threshold=0.10
  global_receiver_mask_rows=2
  global_storage_max_mb=8192
  global_source_regularization=0.10
  correction_csv=FILE           默认 global_gmres_metrics.csv
)KHELP";
    }

    return R"KHELP(
未知程序
========
该可执行文件尚未登记专用帮助。请查看对应源码中的 key=value 参数。
)KHELP";
}

inline bool show_if_requested(int argc, char** argv)
{
    if (!requested(argc, argv)) return false;
    const std::string name =
        executable_name(argc > 0 && argv != nullptr ? argv[0] : "");
    std::cout << program_text(name) << common_footer();
    return true;
}

} // namespace kirch_help

#endif
