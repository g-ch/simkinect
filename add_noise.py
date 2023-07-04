import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage


def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    # Get the size of the dot pattern
    r1, c1 = disp.shape

    # Get a repeated version of the dot pattern
    num_repeats_row = r1 // dot_pattern_.shape[0] + (r1 % dot_pattern_.shape[0] > 0)
    num_repeats_col = c1 // dot_pattern_.shape[1] + (c1 % dot_pattern_.shape[1] > 0)

    repeated_dot_pattern = np.tile(dot_pattern_, (num_repeats_row, num_repeats_col))
    dot_pattern_ = repeated_dot_pattern[0:r1, 0:c1]


    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp



if __name__ == "__main__":

    # reading the image directly in gray with 0 as input 
    dot_pattern_ = cv2.imread("./data/kinect-pattern_3x3.png", 0)

    count = 0

    # various variables to handle the noise modelling
    scale_factor  = 100     # converting depth from m to cm 
    focal_length  = 725.0087 # 480.0   # focal length of the camera used 
    baseline_m    = 0.075   # baseline in m 
    invalid_disp_ = 99999999.9

    max_depth = 400.0

    while count < 447:
        # Set a string for the image sequence number. The number contains 5 digits, 0-padded.
        count_str = str(count).zfill(5)

        depth_uint16 = cv2.imread("/home/clarence/ros_ws/semantic_dsp_ws/src/Semantic_DSP_Map/data/VirtualKitti2/depth/Camera_0/depth_"+count_str+".png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        h, w = depth_uint16.shape 

        # depth in meters 
        depth = depth_uint16.astype('float') / 100.0

        depth_interp = add_gaussian_shifts(depth)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp

        # Let depth larger than max_depth meters be invalid.
        depth[out_disp > max_depth] = 0
        depth[out_disp == invalid_disp_] = 0 
        
        # Axial noise model.
        
        # Option 1: Barron et al. 2013
        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        
        # noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 
        

        # Option 2: Emperrical noise model from Nguyen et al. 2012
        # Calculate standard deviation map. Each pixel has a different standard deviation. The standard deviation is 0.0012+0.0019*(depth-0.4)^2

        std_dev = 0.0012 + 0.0019 * (depth - 0.4)**2

        noise = np.random.normal(0, 1, depth.shape)
        scaled_noise = std_dev * noise

        # Create a mask where depth is greater than 0.4 and less than 600 meters.
        mask = np.logical_and(depth > 0.4, depth < max_depth)
        # Apply noise only to locations where depth is greater than 0.4
        noisy_depth = np.where(mask, depth + scaled_noise, depth)


        noisy_depth = noisy_depth * 100.0 
        noisy_depth = noisy_depth.astype('uint16')

        # Let the depth greater than max_depth_int be 65535.
        max_depth_int = int(max_depth * 100.0)
        noisy_depth[noisy_depth > max_depth_int] = 65535

        # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
        cv2.namedWindow('Adding Kinect Noise', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
        cv2.imshow('Adding Kinect Noise', np.vstack((depth_uint16, noisy_depth)))
        key = cv2.waitKey(100)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        
        out_img_name = "/home/clarence/ros_ws/semantic_dsp_ws/src/Semantic_DSP_Map/data/VirtualKitti2/depth/Camera_0_noised/depth_" + count_str + ".png"
        cv2.imwrite(out_img_name, noisy_depth)

        print(count)
        count = count + 1
