# import hough transformation. Importing numpy here only for creating the point cloud.
from circlehough.hough import main
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time

# simple function to discritize a float
def discritise(value:float) -> int:
    return int(np.round(value))

# disc function to allow for discritizing a tuple
def discritise_tuple(*value:float) -> tuple:
    return tuple([discritise(v) for v in value])

def discritise_array(value:np.ndarray) -> np.ndarray:
    return np.array([discritise(v) for v in value])



if __name__ == '__main__':

    # %% Generate circle, outlier points and point cloud
    # Generate circle and 20 points on it
    SCREEN_SIZE = 200
    r = 50 # radius
    n = 36 # number of points
    t = np.linspace(0, 2*np.pi, n+1)[:-1]
    circle_center_x = discritise(np.random.uniform(SCREEN_SIZE*0.3, SCREEN_SIZE*0.6))
    circle_center_y = discritise(np.random.uniform(SCREEN_SIZE*0.3, SCREEN_SIZE*0.6))
    x = r*np.cos(t) + circle_center_x
    y = r*np.sin(t) + circle_center_y
    # Add some noise to the points
    noise = r * (0.05)
    x += discritise_array(np.random.normal(0, noise, n))
    y += discritise_array(np.random.normal(0, noise, n))

    # remove some points ar random
    chance_to_remove = 0.3
    x, y = zip(*[(x[i], y[i]) for i in range(len(x)) 
                 if np.random.uniform(0, 1) > chance_to_remove])
    
    # Generate outlier points
    center_x = np.random.uniform(discritise(SCREEN_SIZE*0.3), discritise(SCREEN_SIZE*0.6))
    center_y = np.random.uniform(discritise(SCREEN_SIZE*0.3), discritise(SCREEN_SIZE*0.6))
    scale = np.random.uniform(discritise(SCREEN_SIZE/3), discritise(SCREEN_SIZE/3))
    x_outlier = np.random.normal(center_x, scale, 20)
    y_outlier = np.random.normal(center_y, scale, 20)

    # Append outlier points to the original data
    x = np.append(x, x_outlier)
    y = np.append(y, y_outlier)

    # if any x or y is less than 0 or greater than SCREEN_SIZE, remove it
    x, y = zip(*[(x[i], y[i]) for i in range(len(x)) 
                 if x[i] > 0 and x[i] < SCREEN_SIZE and y[i] > 0 and y[i] < SCREEN_SIZE])

    point_cloud = np.column_stack((x, y))

    # Create a currect circle guess
    GUESS_OFF_BY = 0.3
    guessed_cx = discritise(circle_center_x*(1 + np.random.uniform(-GUESS_OFF_BY, GUESS_OFF_BY)))
    guessed_cy = discritise(circle_center_y*(1 + np.random.uniform(-GUESS_OFF_BY, GUESS_OFF_BY)))
    guessed_r = discritise(r*(1 + np.random.uniform(-GUESS_OFF_BY, GUESS_OFF_BY)))

    # uncertainty of the initial guess
    uncertainty_pcnt = 0.8
    uncertainty = guessed_r * (uncertainty_pcnt)

    # width where points can still be counted to be part of the ring
    epsilon_pcnt = 0.5
    epsilon = guessed_r * (epsilon_pcnt)

    
    # %% Plot the data
    
    # plot x y
    plt.figure(figsize=(8, 8))
    plt.xlim(0, SCREEN_SIZE)
    plt.ylim(0, SCREEN_SIZE)
    plt.scatter(x, y, color='white', label='Point cloud')

    # Draw real center with red
    plt.scatter(circle_center_x, circle_center_y, color='red', label='Real center')
    # Draw real circle with red
    circle = plt.Circle((circle_center_x, circle_center_y), r, color='r', fill=False, label='Real circle')
    plt.gca().add_artist(circle)

    # Draw guessed center with yellow
    plt.scatter(guessed_cx, guessed_cy, color='yellow', label='Guessed center')
    # Draw guessed circle with yellow
    circle = plt.Circle((guessed_cx, guessed_cy), guessed_r, color='yellow', fill=False, label='Guessed circle')
    plt.gca().add_artist(circle)

    # Draw guessed circle + uncertainty area with .25 alpha
    # uncertainty area will be circle thickenss on plot
    circle = plt.Circle((guessed_cx, guessed_cy), guessed_r, color='yellow', alpha=0.25, fill=False, linewidth=uncertainty)
    plt.gca().add_artist(circle)



    # %% Perform the transformation

    # initial guess for the ring center and radius (if no previous info about those, increase uncertainty accordingly)
    # guessed_cx = 0.1
    # guessed_cy = 0.1
    # guessed_r = 1.9


    # Start the timer
    start_time = time.time()

    # perform the transformation
    hough_cx, hough_cy, hough_r = main(
        guessed_cx, guessed_cy, guessed_r, point_cloud,
        uncertainty, epsilon
    )

    # Calculate the execution time
    execution_time = time.time() - start_time

    # %% Plot the result
    
    # plot x y of the found circle
    plt.scatter(hough_cx, hough_cy, color='green', label='Found center')

    # plot the found circle
    circle = plt.Circle((hough_cx, hough_cy), hough_r, color='green', fill=False, label='Found circle')
    plt.gca().add_artist(circle)
    circle = plt.Circle((hough_cx, hough_cy), hough_r, color='green', alpha=0.25, fill=False, linewidth=discritise(epsilon))
    plt.gca().add_artist(circle)

    # Find all inliers using the found circle and epsilon
    inliers = []
    outliers = []
    for point in zip(x, y):
        distance = np.sqrt((point[0] - hough_cx)**2 + (point[1] - hough_cy)**2)
        if abs(distance - hough_r) < epsilon:
            inliers.append(point)
        else:
            outliers.append(point)
    inliers, outliers = np.array(inliers), np.array(outliers)

    # Draw small green circles in inliers
    plt.scatter(inliers[:, 0], inliers[:, 1], color='green', s=10, label='Found inliers')
    # Draw small red circles in outliers
    plt.scatter(outliers[:, 0], outliers[:, 1], color='red', s=10, label='Found outliers')




    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

    # %% Print stats

    print(f"Execution time: {execution_time*1000:.0f} ms")

    # Circle Error as percentage of original circle radius
    circle_error = abs(hough_r - r)/r
    print(f"Circle error: {circle_error:.2%}")

    # Center Error as percentage of original circle radius
    center_error = np.sqrt((hough_cx - circle_center_x)**2 + (hough_cy - circle_center_y)**2)/r
    print(f"Center error: {center_error:.2%}")



    
