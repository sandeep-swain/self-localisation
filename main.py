import pyrealsense2 as rs
import numpy as np
import cv2
import math
from mpl_toolkits import mplot3d
#graph
import matplotlib.pyplot as plt
i = 0
x_co =[0]
y_co =[0]
z_co =[0]

#details of camera used
global fov_x
global fov_y
fov_x=1.50098
fov_y=0.99483
m_px=320.0
m_py=240.0

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)




#from dataset
def pixel_den_x(d):
    depth=d
    total_pixel_x=640
    fx = 600.0
    cx = 320.0
    #fov_x=2*(math.atan(cx/(2*fx)))
    #fov_x=1.50098
    #print('fovx'+str(fov_x))
    cpp_x=(2*depth*math.tan(fov_x/2)/total_pixel_x)
    return cpp_x


def pixel_den_y(d):
    depth=d
    total_pixel_y=480
    fy = 600.0
    cy = 240.0
    #fov_y=2*(math.atan(cy/(2*fy)))
    #fov_y=0.99483
    #print('fovy'+str(fov_y))
    cpp_y=(2*depth*math.tan(fov_y/2)/total_pixel_y)
    return cpp_y



feature_params = dict(maxCorners = 300, qualityLevel = 0.02, minDistance = 2, blockSize = 7)
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#cap = cv.VideoCapture(0)

color = (0, 255, 0)
output_x=0
output_y=0
count=0
output_z=0

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
   

        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        print('max')
        print(np.amax(depth_image))
        print(depth_image[222][222])

        color_image = np.asanyarray(color_frame.get_data())
        
       
        #print(depth_image.shape)
        #print(depth_image[0][200])
        total=0
        if count<100:
            prev_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        count=len(prev)


        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        good_old = prev[status == 1]
        good_new = next[status == 1]

        sum_x=0
        sum_y=0
        sum_z=0


        for (new, old) in zip(good_new, good_old):

            a, b = new.ravel()
            c, d = old.ravel()

            if (c<20 or d>460 or d<20 or c> 620):
                continue
            if (a<20 or b>460 or b<20 or a> 620):
                continue


            dop_x=320/math.tan(fov_x/2)
            dop_y=240/math.tan(fov_y/2)    


            del_x=abs(a-m_px)
            del_y=abs(b-m_py)
            tita_y=(del_y/dop_y)
            tita_x=(del_x/dop_x)

            t_x_sq=(tita_x*tita_x)
            t_y_sq=(tita_y*tita_y)
            tan_tita=math.sqrt(t_y_sq+t_x_sq)
            tita=math.atan(tan_tita)
            #print('o')
            #print(c,d)
            #print('n')
            #print(a+1,b+1)
            pixel_val=depth_image[(int(d),int(c))]*math.cos(tita)
            r_pixel_val=depth_image[(int(b),int(a))]*math.cos(tita)
            if(pixel_val>6000 or r_pixel_val>6000):
                continue
            #print(pixel_val)

            cmpp_x=pixel_den_x(pixel_val/20)
            distance_x=(a-c)*cmpp_x
            #print('DISTANCE_x:- '+str(distance_x))

            cmpp_y=pixel_den_y(pixel_val/10)
            distance_y=(d-b)*cmpp_y
            # print('DISTANCE_y:- '+str(distance_y))

            distance_z=(r_pixel_val/20)-(pixel_val/20)
            
            #print('distance_z '+str(distance_z))
            sum_z=sum_z+distance_z
        
            
        
        
            sum_x=sum_x+distance_x
            sum_y=sum_y+distance_y
            # mask = cv.line(mask, (a, b), (c, d), color, 2)
            color_image = cv2.circle(color_image, (a, b), 3, color, -1)
            total=total+1


        #print(total)
        

    
    


        dist_x=(sum_x/total)
        output_x=output_x+dist_x

        dist_z=(sum_z/total)
        output_z=output_z+dist_z

        dist_y=(sum_y/total)
        output_y=output_y+dist_y

        
        print('DISTANCE_x:- '+str(output_x))
        print('DISTANCE_y:- '+str(output_y))
        print('DISTANCE_z:- '+str(output_z))

        x_co.append(-output_x)
        z_co.append(output_z)
        y_co.append(-output_y)


        #print('total '+str(total))


        #output = cv.add(frame, mask)
 
        prev_gray = gray.copy()
 
        prev = good_new.reshape(-1, 1, 2)


       
        cv2.imshow('depth',depth_image)
        cv2.imshow('color',color_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            
            break

finally:

    # Stop streaming
    pipeline.stop()

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(x_co, y_co, z_co, 'green') 



plt.show()
