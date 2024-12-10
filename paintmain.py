import pygame
import numpy as np
from naivebayes import downsample_to_28x28_nonsquare
from naivebayes import BNB
import pandas as pd

# left click to draw, right click to clear
# press enter to confirm selection once you finish drawing the number that you want to estimate 

isRunning = True
prev_mouse_pos = None
canvas = np.zeros((360, 400), dtype=np.uint8)  # Canvas for drawing (grayscale image)

# Initialize extreme points (bounding box edges)
leftmost = 360 
rightmost = 0
topmost = 400
bottommost = 0

# Initialize pygame
pygame.init()
pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
my_font = pygame.font.SysFont('Monaco', 30)
window = pygame.display.set_mode((360, 400))
pygame.display.set_caption("left click to draw, right click to clear")
clock = pygame.time.Clock()

# Surface for numbers
num_surface = pygame.Surface(window.get_size())
num_surface.fill((0, 0, 0))

# Animation variables
flash = False
animating = False
animation_timer = 0
animation_step = 0

# global animating, animation_timer, animation_step
# Update the bounding box edges based on a new point (x, y)
def update_bounding_box(x, y):
    global leftmost, rightmost, topmost, bottommost
    leftmost = min(leftmost, x)
    rightmost = max(rightmost, x)
    topmost = min(topmost, y)
    bottommost = max(bottommost, y)

# Draw the bounding box on the surface.
def draw_bounding_box(surface, color, rect, rect_width):
    if leftmost < rightmost and topmost < bottommost:  # Ensure valid bounding box
        # Create a semi-transparent surface
        width = rightmost - leftmost 
        height = bottommost - topmost 
        shape_surf = pygame.Surface((width + 20, height + 20), pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), width=rect_width)
        surface.blit(shape_surf, (leftmost - 10, topmost - 10))

# Play the bounding box flashing animation.
def play_confirmed_animation(guessed_number):
    global animating, animation_timer, animation_step

    # Calculate bounding box dimensions
    width = rightmost - leftmost
    height = bottommost - topmost
    rect = (leftmost + 10, topmost + 10, width + 10, height + 10)

    # Flashing sequence: alternates between filled and outlined
    animation_colors = [
        (0, 255, 0, 127),    # Cyan
        (0, 255, 0, 127),    # Red
        (0, 255, 0, 127),    # Green
        (0, 255, 0, 127),
        (0, 255, 0, 127)
    ]
    if animating:
        if pygame.time.get_ticks() - animation_timer > 100:  # Change every 100ms
            animation_timer = pygame.time.get_ticks()
            animation_step = (animation_step + 1) % len(animation_colors)

        # Alternate between filled and outline
        if animation_step % 2 == 0:
            draw_bounding_box(window, animation_colors[animation_step], rect, 0)
        else:
            draw_bounding_box(window, animation_colors[animation_step], rect, 3)

        # End animation after a few flashes
        if animation_step == len(animation_colors) - 1:
            play_guess_result(guessed_number)
            animating = False

# switches screen to say "model's guess: {number}"
# for a couple of seconds, and then switches back to number screen
def play_guess_result(guessed_number):
    sceneExit = False
    time = 3000  # 2000 milliseconds until we continue.

    while not sceneExit:
        window.fill((0,0,0))
        text_surface = my_font.render(f"Model guessed: {guessed_number}", False, (255, 255, 255))
        window.blit(text_surface, (0,0))
        pygame.display.update()

        passed_time = clock.tick(30)
        time -= passed_time
        if time <= 0:
            sceneExit = True
            
#### INITIALIZING MODEL ####

test_data = pd.read_csv("mnist_test_final.csv")
train_data = pd.read_csv("mnist_train_final.csv")

x_test = test_data.drop(columns = ['label'])
y_test = test_data['label']
x_train = train_data.drop(columns = ['label'])
y_train = train_data['label']

# scale the data to be between 0 and 1 
x_test = x_test / 255
x_train = x_train / 255

# binarize the mnist data 
def binarize(data):
    # Use NumPy for element-wise operations, which is faster and more concise
    return (data > 0).astype(int)

x_train_binary = binarize(x_train)
x_test_binary = binarize(x_test)

model = BNB()
# Fit the model
priors, likelihoods = model.fit(x_train_binary, y_train)

# when user presses enter, reduce the canvas to a 28x28, then convert it to a 1-d array,
# then plug into naive-bayes model, get prediction, and output result of prediction to window
def get_prediction(canvas):
    # Ensure bounding box is valid
    if rightmost <= leftmost or bottommost <= topmost:
        print("No valid drawing detected.")
        return None

    # Crop the canvas to the bounding box
    cropped_canvas = canvas[topmost - 10:bottommost + 10, leftmost - 10:rightmost + 10]
    np.save('bounding_box_canvas.npy', cropped_canvas)    

    downscaled_canvas = downsample_to_28x28_nonsquare(cropped_canvas)
    vectorized_canvas = downscaled_canvas.flatten()  # Convert to 1D
    vectorized_canvas = vectorized_canvas.reshape(1, -1)
    #pred_class = model.predict(vectorized_canvas.reshape(1, -1))
    pred_class = model.predict(vectorized_canvas)
    return pred_class

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


######################################################
###########    START OF MAIN PYGAME LOOP   ###########
######################################################
while isRunning:
    # Poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and not animating:
                animating = True
                animation_timer = pygame.time.get_ticks()
                animation_step = 0

    # Determine if left click is pressed
    pygame.event.get()
    left_click_pressed = pygame.mouse.get_pressed(num_buttons=3)[0]
    right_click_pressed = pygame.mouse.get_pressed(num_buttons=3)[2]

    # Determine if mouse is on the screen
    mouse_pos = pygame.mouse.get_pos()

    # Draw from mouse input
    if pygame.mouse.get_focused() and left_click_pressed:
        if prev_mouse_pos is None:
            pygame.draw.rect(window, (255, 255, 255), pygame.Rect((mouse_pos[0], mouse_pos[1]), (3, 3)), 0)
            pygame.draw.rect(num_surface, (255, 255, 255), pygame.Rect((mouse_pos[0], mouse_pos[1]), (3, 3)), 0)
            # update canvas to reflect changes
            # Update canvas for rectangle
            x, y = mouse_pos
            #canvas[y, x] = 1
            canvas[max(0, y-1):min(canvas.shape[0], y+2), max(0, x-1):min(canvas.shape[1], x+2)] = 1
            np.save('canvas.npy', canvas)
        else:
            pygame.draw.line(window, (255, 255, 255), prev_mouse_pos, mouse_pos, 3)
            pygame.draw.line(num_surface, (255, 255, 255), prev_mouse_pos, mouse_pos, 3)
            # update canvas to reflect changes 
            # Update canvas for line
            x0, y0 = prev_mouse_pos
            x1, y1 = mouse_pos
            # Use bresenham's line algorithm to get all points in the line
            for x, y in bresenham_line(x0, y0, x1, y1):
                canvas[max(0, y-1):min(canvas.shape[0], y+2), max(0, x-1):min(canvas.shape[1], x+2)] = 1
                #canvas[y0, x0] = 1
                #canvas[y1, x1] = 1
                np.save('canvas.npy', canvas)
                
        update_bounding_box(mouse_pos[0], mouse_pos[1])

    # Update previous mouse position
    prev_mouse_pos = mouse_pos if left_click_pressed else None

    # Clear screen if right click is pressed and erase canvas
    if right_click_pressed:
        #np.save('canvas.npy', canvas)
        window.fill((0, 0, 0))
        num_surface.fill((0, 0, 0))
        leftmost, rightmost, topmost, bottommost = 360, 0, 400, 0
        # reset canvas 
        # Save the canvas to a .npy file
        canvas = np.zeros((360, 400), dtype=np.uint8)

    window.blit(num_surface, (0, 0))  # Draw the number surface

    # Draw bounding box
    width = rightmost - leftmost
    height = bottommost - topmost
    rect = (leftmost + 10, topmost + 10, width + 10, height + 10)

    if animating:
        play_confirmed_animation(get_prediction(canvas))
        #play_confirmed_animation(3)
    else:
        draw_bounding_box(window, (0, 0, 255), rect, 3)  # Default bounding box

    # Update the display
    pygame.display.flip()

    # Cap the frame rate to 30 FPS
    clock.tick(30)

pygame.quit()

######################################################
###########    END OF MAIN PYGAME LOOP  ##############
######################################################
