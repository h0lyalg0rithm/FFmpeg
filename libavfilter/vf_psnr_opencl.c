/*
 * Copyright (c) 2011 Roger Pau Monné <roger.pau@entel.upc.edu>
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2013 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Calculate the PSNR between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/file_open.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "drawutils.h"
#include "framesync.h"
#include "internal.h"
#include "psnr.h"
#include "opencl.h"
#include "opencl_source.h"
#include "time.h"

typedef struct PSNROpenCLContext
{
    OpenCLFilterContext ocf;
    int initialised;
    FFFrameSync fs;
    cl_kernel kernel;
    cl_mem mse_img;
    cl_mem wm;
    cl_kernel sum;
    cl_command_queue command_queue;
    double mse, min_mse, max_mse, mse_comp[4];
    uint64_t nb_frames;
    FILE *stats_file;
    char *stats_file_str;
    int stats_version;
    int stats_header_written;
    int stats_add_max;
    int max[4], average_max;
    int is_rgb;
    uint8_t rgba_map[4];
    char comps[4];
    int nb_components;
    int planewidth[4];
    int planeheight[4];
    double planeweight[4];
    uint64_t *score;
    int counter;
} PSNROpenCLContext;

int ff_framesync_dualinput_get_simple(FFFrameSync *fs, AVFrame **f0, AVFrame **f1);
int ff_framesync_dualinput_get_simple(FFFrameSync *fs, AVFrame **f0, AVFrame **f1)
{
    AVFilterContext *ctx = fs->parent;
    AVFrame *mainpic = NULL, *secondpic = NULL;
    int ret;

    if ((ret = ff_framesync_get_frame(fs, 0, &mainpic, 0)) < 0 ||
        (ret = ff_framesync_get_frame(fs, 1, &secondpic, 0)) < 0)
    {
        return ret;
    }
    av_assert0(mainpic);
    av_assert0(secondpic);
    mainpic->pts = av_rescale_q(fs->pts, fs->time_base, ctx->outputs[0]->time_base);
    secondpic->pts = av_rescale_q(fs->pts, fs->time_base, ctx->outputs[0]->time_base);
    if (ctx->is_disabled)
        secondpic = NULL;
    *f0 = mainpic;
    *f1 = secondpic;
    return 0;
}
static inline unsigned pow_2(unsigned base)
{
    return base * base;
}

static inline double get_psnr(double mse, uint64_t nb_frames, int max)
{
    return 10.0 * log10(pow_2(max) / (mse / nb_frames));
}

static int psnr_opencl_load(AVFilterContext *avctx,
                            enum AVPixelFormat main_format,
                            enum AVPixelFormat ref_format)
{
    PSNROpenCLContext *ctx = avctx->priv;
    int err, sum, j;
    const char *source = ff_opencl_source_psnr;
    cl_int cle;
    AVFilterLink *inlink = avctx->inputs[0];
    double average_max;
    const AVPixFmtDescriptor *main_desc;
    cl_int height, width;
    cl_image_format img_format;
    cl_image_desc img_desc;

    err = ff_opencl_filter_load_program(avctx, &source, 1);
    if (err < 0)
        goto fail;

    main_desc = av_pix_fmt_desc_get(main_format);
    ctx->nb_components = main_desc->nb_components;

    if (avctx->inputs[0]->w != avctx->inputs[1]->w ||
        avctx->inputs[0]->h != avctx->inputs[1]->h)
    {
        av_log(avctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }

    ctx->max[0] = (1 << main_desc->comp[0].depth) - 1;
    ctx->max[1] = (1 << main_desc->comp[1].depth) - 1;
    ctx->max[2] = (1 << main_desc->comp[2].depth) - 1;
    ctx->max[3] = (1 << main_desc->comp[3].depth) - 1;

    ctx->is_rgb = ff_fill_rgba_map(ctx->rgba_map, main_format) >= 0;
    ctx->comps[0] = ctx->is_rgb ? 'r' : 'y';
    ctx->comps[1] = ctx->is_rgb ? 'g' : 'u';
    ctx->comps[2] = ctx->is_rgb ? 'b' : 'v';
    ctx->comps[3] = 'a';

    ctx->planeheight[1] = ctx->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, main_desc->log2_chroma_h);
    ctx->planeheight[0] = ctx->planeheight[3] = inlink->h;
    ctx->planewidth[1] = ctx->planewidth[2] = AV_CEIL_RSHIFT(inlink->w, main_desc->log2_chroma_w);
    ctx->planewidth[0] = ctx->planewidth[3] = inlink->w;
    sum = 0;
    for (j = 0; j < ctx->nb_components; j++)
        sum += ctx->planeheight[j] * ctx->planewidth[j];
    average_max = 0;
    for (j = 0; j < ctx->nb_components; j++)
    {
        ctx->planeweight[j] = (double)ctx->planeheight[j] * ctx->planewidth[j] / sum;
        average_max += ctx->max[j] * ctx->planeweight[j];
    }
    ctx->average_max = lrint(average_max);

    ctx->score = av_calloc(ctx->nb_components, sizeof(*ctx->score));
    if (!ctx->score)
        return AVERROR(ENOMEM);

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              CL_QUEUE_PROFILING_ENABLE, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create OpenCL "
                                   "command queue %d.\n",
                     cle);

    av_log(avctx, AV_LOG_DEBUG, "Setting up kernel .\n");

    ctx->kernel = clCreateKernel(ctx->ocf.program, "compute_images_mse_8bit", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel compute_images_mse_8bit %d.\n", cle);

    ctx->sum = clCreateKernel(ctx->ocf.program, "sum", &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create kernel sum %d\n", cle);

    height = ctx->planeheight[0];
    width = ctx->planewidth[0];

    img_format = (cl_image_format){CL_RGBA, CL_SIGNED_INT32};

    img_desc = (cl_image_desc){
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = width,
        .image_height = height,
        .image_depth = 0,
        .image_array_size = 0,
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        .buffer = NULL,
    };

    ctx->mse_img = clCreateImage(ctx->ocf.hwctx->context, 0,
                                 &img_format, &img_desc, NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create buffer: %d.\n", cle);

    ctx->wm = clCreateBuffer(ctx->ocf.hwctx->context, 0, 4 * height * sizeof(cl_ulong), NULL, &cle);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to create buffer: %d.\n", cle);

    ctx->initialised = 1;
    return 0;

fail:
    if (ctx->command_queue)
        clReleaseCommandQueue(ctx->command_queue);
    if (ctx->kernel)
        clReleaseKernel(ctx->kernel);
    return err;
}

#define OFFSET(x) offsetof(PSNROpenCLContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

static const AVOption psnr_opencl_options[] = {
    {"stats_file", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"f", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, FLAGS},
    {"stats_version", "Set the format version for the stats file.", OFFSET(stats_version), AV_OPT_TYPE_INT, {.i64 = 1}, 1, 2, FLAGS},
    {"output_max", "Add raw stats (max values) to the output log.", OFFSET(stats_add_max), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, FLAGS},
    {NULL}};

FRAMESYNC_DEFINE_CLASS(psnr_opencl, PSNROpenCLContext, fs);

static void set_meta(AVDictionary **metadata, const char *key, char comp, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%f", d);
    if (comp)
    {
        char key2[128];
        snprintf(key2, sizeof(key2), "%s%c", key, comp);
        av_dict_set(metadata, key2, value, 0);
    }
    else
    {
        av_dict_set(metadata, key, value, 0);
    }
}

static int do_opencl_psnr(FFFrameSync *fs)
{
    AVFilterContext *avctx = fs->parent;
    PSNROpenCLContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    AVFrame *output;
    AVFrame *master, *ref;
    int ret, err, i;
    cl_int cle;
    size_t global_work[2];
    int plane, kernel_arg = 0;
    cl_mem main_mem, ref_mem;
    cl_ulong *data_mem;
    double psnr_avg;
    uint64_t comp_sum[4] = {0};
    double comp_mse[4], mse = 0.;
    AVDictionary **metadata;
    cl_int height, width;

    ret = ff_framesync_dualinput_get_simple(fs, &master, &ref);
    if (ret < 0)
        return ret;

    if (avctx->is_disabled || !ref)
        return ff_filter_frame(outlink, master);

    if (!ctx->initialised)
    {
        AVHWFramesContext *master_fc =
            (AVHWFramesContext *)master->hw_frames_ctx->data;
        AVHWFramesContext *ref_fc =
            (AVHWFramesContext *)ref->hw_frames_ctx->data;

        err = psnr_opencl_load(avctx, master_fc->sw_format,
                               ref_fc->sw_format);
        if (err < 0)
            return err;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    i = 0;

    kernel_arg = 0;
    height = ctx->planeheight[i];
    width = ctx->planewidth[i];

    main_mem = (cl_mem)master->data[i];
    CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &main_mem);
    kernel_arg++;

    ref_mem = (cl_mem)ref->data[i];
    CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &ref_mem);
    kernel_arg++;

    CL_SET_KERNEL_ARG(ctx->kernel, kernel_arg, cl_mem, &ctx->mse_img);
    kernel_arg++;

    err = ff_opencl_filter_work_size_from_image(avctx, global_work,
                                                master, i, 0);

    if (err < 0)
        goto fail;

    av_log(avctx, AV_LOG_DEBUG, "Enqueing kernel\n");
    cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->kernel, 2, NULL,
                                 global_work, NULL, 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to enqueue psnr kernel "
                                   "for plane %d: %d.\n",
                     plane, cle);

    CL_SET_KERNEL_ARG(ctx->sum, 0, cl_mem, &ctx->mse_img);
    CL_SET_KERNEL_ARG(ctx->sum, 1, cl_int, &height);
    CL_SET_KERNEL_ARG(ctx->sum, 2, cl_int, &width);
    CL_SET_KERNEL_ARG(ctx->sum, 3, cl_mem, &ctx->wm);

    size_t worksize[2] = {height, 1};
    cle = clEnqueueNDRangeKernel(ctx->command_queue, ctx->sum, 1, NULL,
                                 worksize, NULL, 0, NULL, NULL);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to get data: %d.\n", cle);

    data_mem = av_malloc(4 * height * sizeof(cl_ulong));
    if (!data_mem)
        return AVERROR(ENOMEM);

    cle = clEnqueueReadBuffer(ctx->command_queue, ctx->wm, CL_FALSE, 0, 4 * height * sizeof(cl_ulong), data_mem, 0, NULL, NULL);

    cle = clFinish(ctx->command_queue);
    CL_FAIL_ON_ERROR(AVERROR(EIO), "Failed to finish command queue: %d.\n", cle);

    metadata = &master->metadata;

    for (int i = 0; i < 4 * height; i += 4)
    {
        comp_sum[0] += data_mem[i + 0];
        comp_sum[1] += data_mem[i + 1];
        comp_sum[2] += data_mem[i + 2];
        comp_sum[3] += data_mem[i + 3];
    }

    for (int c = 0; c < ctx->nb_components; c++)
        comp_mse[c] = comp_sum[c] / ((double)ctx->planewidth[c] * ctx->planeheight[c]);

    for (int c = 0; c < ctx->nb_components; c++)
        mse += comp_mse[c] * ctx->planeweight[c];

    av_log(ctx, AV_LOG_TRACE, "values %f %f %f %f\n ", comp_mse[0], comp_mse[1], comp_mse[2], comp_mse[3]);

    ctx->min_mse = FFMIN(ctx->min_mse, mse);
    ctx->max_mse = FFMAX(ctx->max_mse, mse);

    ctx->mse += mse;

    for (int j = 0; j < ctx->nb_components; j++)
        ctx->mse_comp[j] += comp_mse[j];
    ctx->nb_frames++;

    av_log(ctx, AV_LOG_TRACE, "values %f %f %f %f\n ", comp_mse[0], comp_mse[1], comp_mse[2], comp_mse[3]);

    for (int j = 0; j < ctx->nb_components; j++)
    {
        int c = ctx->is_rgb ? ctx->rgba_map[j] : j;
        set_meta(metadata, "lavfi.psnr.mse.", ctx->comps[j], comp_mse[c]);
        set_meta(metadata, "lavfi.psnr.psnr.", ctx->comps[j], get_psnr(comp_mse[c], 1, ctx->max[c]));
    }
    set_meta(metadata, "lavfi.psnr.mse_avg", 0, mse);
    psnr_avg = get_psnr(mse, 1, ctx->average_max);
    set_meta(metadata, "lavfi.psnr.psnr_avg", 0, psnr_avg);

    if (ctx->stats_file)
    {
        if (ctx->stats_version == 2 && !ctx->stats_header_written)
        {
            fprintf(ctx->stats_file, "psnr_log_version:2 fields:n");
            fprintf(ctx->stats_file, ",mse_avg");
            for (int j = 0; j < ctx->nb_components; j++)
            {
                fprintf(ctx->stats_file, ",mse_%c", ctx->comps[j]);
            }
            fprintf(ctx->stats_file, ",psnr_avg");
            for (int j = 0; j < ctx->nb_components; j++)
            {
                fprintf(ctx->stats_file, ",psnr_%c", ctx->comps[j]);
            }
            if (ctx->stats_add_max)
            {
                fprintf(ctx->stats_file, ",max_avg");
                for (int j = 0; j < ctx->nb_components; j++)
                {
                    fprintf(ctx->stats_file, ",max_%c", ctx->comps[j]);
                }
            }
            fprintf(ctx->stats_file, "\n");
            ctx->stats_header_written = 1;
        }
        fprintf(ctx->stats_file, "n:%" PRId64 " mse_avg:%0.2f ", ctx->nb_frames, mse);
        for (int j = 0; j < ctx->nb_components; j++)
        {
            int c = ctx->is_rgb ? ctx->rgba_map[j] : j;
            fprintf(ctx->stats_file, "mse_%c:%0.2f ", ctx->comps[j], comp_mse[c]);
        }
        fprintf(ctx->stats_file, "psnr_avg:%0.2f ", psnr_avg);
        for (int j = 0; j < ctx->nb_components; j++)
        {
            int c = ctx->is_rgb ? ctx->rgba_map[j] : j;
            fprintf(ctx->stats_file, "psnr_%c:%0.2f ", ctx->comps[j],
                    get_psnr(comp_mse[c], 1, ctx->max[c]));
        }
        if (ctx->stats_version == 2 && ctx->stats_add_max)
        {
            fprintf(ctx->stats_file, "max_avg:%d ", ctx->average_max);
            for (int j = 0; j < ctx->nb_components; j++)
            {
                int c = ctx->is_rgb ? ctx->rgba_map[j] : j;
                fprintf(ctx->stats_file, "max_%c:%d ", ctx->comps[j], ctx->max[c]);
            }
        }
        fprintf(ctx->stats_file, "\n");
        fflush(ctx->stats_file);
    }

    av_free(data_mem);
    return ff_filter_frame(outlink, output);

fail:
    return err;
}

static av_cold int psnr_opencl_init(AVFilterContext *ctx)
{
    PSNROpenCLContext *s = ctx->priv;

    s->min_mse = +INFINITY;
    s->max_mse = -INFINITY;

    if (s->stats_file_str)
    {
        if (s->stats_version < 2 && s->stats_add_max)
        {
            av_log(ctx, AV_LOG_ERROR,
                   "stats_add_max was specified but stats_version < 2.\n");
            return AVERROR(EINVAL);
        }
        if (!strcmp(s->stats_file_str, "-"))
        {
            s->stats_file = stdout;
        }
        else
        {
            s->stats_file = avpriv_fopen_utf8(s->stats_file_str, "w");
            if (!s->stats_file)
            {
                int err = AVERROR(errno);
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                av_log(ctx, AV_LOG_ERROR, "Could not open stats file %s: %s\n",
                       s->stats_file_str, buf);
                return err;
            }
        }
    }

    s->fs.on_event = do_opencl_psnr;
    return ff_opencl_filter_init(ctx);
}

static int psnr_opencl_config_input_ref(AVFilterLink *inlink)
{
    AVFilterContext *avctx = inlink->dst;

    if (!inlink->hw_frames_ctx)
    {
        av_log(avctx, AV_LOG_ERROR, "OpenCL filtering requires a "
                                    "hardware frames context on the input.\n");
        return AVERROR(EINVAL);
    }

    if (avctx->inputs[1] != inlink)
        return 0;

    return ff_opencl_filter_config_input(inlink);
}

static int psnr_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    PSNROpenCLContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    AVFilterLink *reflink = ctx->inputs[1];
    int ret;

    ret = ff_framesync_init_dualinput(&s->fs, ctx);
    if (ret < 0)
        return ret;
    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_framesync_configure(&s->fs)) < 0)
        return ret;

    outlink->time_base = s->fs.time_base;

    if (av_cmp_q(mainlink->time_base, outlink->time_base) ||
        av_cmp_q(reflink->time_base, outlink->time_base))
        av_log(ctx, AV_LOG_WARNING, "not matching timebases found between first input: %d/%d and second input %d/%d, results may be incorrect!\n",
               mainlink->time_base.num, mainlink->time_base.den,
               reflink->time_base.num, reflink->time_base.den);

    return ff_framesync_configure(&s->fs);
}

static int psnr_opencl_activate(AVFilterContext *ctx)
{
    PSNROpenCLContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static av_cold void psnr_opencl_uninit(AVFilterContext *ctx)
{
    PSNROpenCLContext *s = ctx->priv;

    if (s->nb_frames > 0)
    {
        int j;
        char buf[256];

        buf[0] = 0;
        for (j = 0; j < s->nb_components; j++)
        {
            int c = s->is_rgb ? s->rgba_map[j] : j;
            av_strlcatf(buf, sizeof(buf), " %c:%f", s->comps[j],
                        get_psnr(s->mse_comp[c], s->nb_frames, s->max[c]));
        }
        av_log(ctx, AV_LOG_INFO, "PSNR%s average:%f min:%f max:%f\n",
               buf,
               get_psnr(s->mse, s->nb_frames, s->average_max),
               get_psnr(s->max_mse, 1, s->average_max),
               get_psnr(s->min_mse, 1, s->average_max));
    }

    ff_framesync_uninit(&s->fs);
    clReleaseMemObject(s->mse_img);
    clReleaseMemObject(s->wm);
    av_freep(&s->score);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);
}

static const AVFilterPad psnr_opencl_inputs[] = {
    {
        .name = "main",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_input,
    },
    {
        .name = "reference",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = &psnr_opencl_config_input_ref,
    },
};

static const AVFilterPad psnr_opencl_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = psnr_opencl_config_output,
    },
};

const AVFilter ff_vf_psnr_opencl = {
    .name = "psnr_opencl",
    .description = NULL_IF_CONFIG_SMALL("Calculate the PSNR between two video streams."),
    .preinit = psnr_opencl_framesync_preinit,
    .init = psnr_opencl_init,
    .uninit = psnr_opencl_uninit,
    .activate = psnr_opencl_activate,
    .priv_size = sizeof(PSNROpenCLContext),
    .priv_class = &psnr_opencl_class,
    FILTER_INPUTS(psnr_opencl_inputs),
    FILTER_OUTPUTS(psnr_opencl_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_OPENCL),
    .flags = FF_FILTER_FLAG_HWFRAME_AWARE |
             AVFILTER_FLAG_METADATA_ONLY,
};
