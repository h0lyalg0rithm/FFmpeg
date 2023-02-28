__kernel void compute_images_mse_8bit(__read_only  image2d_t main,
                                      __read_only  image2d_t ref,
                                      __write_only image2d_t mse_img){
  const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_FILTER_NEAREST);
  int2 loc = (int2)(get_global_id(0), get_global_id(1));
  int msex = 0, msey =  0, msez = 0;

  float4 main_f, ref_f;

  main_f = read_imagef(main, sampler, loc);
  ref_f = read_imagef(ref, sampler, loc);
  msex = ((main_f.x - ref_f.x) * (main_f.x - ref_f.x) * 255 * 255);
  msey = ((main_f.y - ref_f.y) * (main_f.y - ref_f.y) * 255 * 255);
  msez = ((main_f.z - ref_f.z) * (main_f.z - ref_f.z) * 255 * 255);
  int4 x = (int4)(msex, msey, msez, 0.0);
  write_imagei(mse_img, loc, x);
  
  
}

__kernel void sum(__read_only image2d_t mse_img,
                  int height,
                  int width,
                  global ulong4* result){
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_FILTER_NEAREST);
    int index = get_global_id(0);
    ulong4 sum = 0;

    if(index > height)
    return;
    for(int w = 0; w < width; w++){
        int4 r = read_imagei(mse_img, sampler, (int2)(w, index));
        sum.x += r.x;
        sum.y += r.y;
        sum.z += r.z;
    }
    sum.w = 0;
    result[index] = sum;
}
