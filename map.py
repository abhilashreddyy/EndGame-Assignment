# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from aiT3D import TD3, ReplayBuffer
import random
import cv2
from scipy import ndimage
from PIL import Image
import scipy


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1509')
Config.set('graphics', 'height', '740')


# model parameters START
seed = 0 # Random seed number
start_timesteps = 9e2 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e2 # How often the evaluation step is performed (after how many timesteps)
#max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 1.0 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 30 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
done = True
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_reward = 0
episode_timesteps = 0
reached_dest = 0

action_len = 1
state_len = 5
last_time_steps = 1
image_size = 60
orientation = -0.9
#obs = [0.23,1,1,0.5, -0.5]
# model parameters END

# model global params
replay_buffer = ReplayBuffer()
# model global params


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
max_action_agent = 40
brain = TD3(state_len,action_len,max_action_agent)
action2rotation = [0,5,-5]
reward = 0
scores = []
reward_window = []
im = CoreImage("./images/MASK1_white.png")
main_img = cv2.imread('./images/mask_white.png',0)

# textureMask = CoreImage(source="./kivytest/simplemask1.png")
def save_cropped_image(img, x, y, name = ""):
    # print("entered")
    # data = np.array(img)# * 255.0
    # rescaled = data.astype(np.uint8)
    # im = Image.fromarray(rescaled)
    # im.save("./check/"+name+ "_" + "your_file"+str(x) +"_"+ str(y) +".png")
    return

def get_target_image(img, angle, center, size, fill_with = 100.0):
    angle = angle + 90
    center[0] -= 0
    img = np.pad(img, size, 'constant', constant_values = fill_with)
    init_size = 1.6*size
    center[0] += size
    center[1] += size
    cropped = img[int(center[0]-(init_size/2)) : int(center[0]+(init_size/2)) ,int(center[1]-(init_size/2)): int(center[1]+(init_size/2))]
    rotated = ndimage.rotate(cropped, angle, reshape = False, cval = 255.0)
    y,x = rotated.shape
    final = rotated[int(y/2-(size/2)):int(y/2+(size/2)),int(x/2-(size/2)):int(x/2+(size/2))]
    return cropped, rotated, final


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask_white.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1460
    goal_y = 662
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation



# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
#        my_rand_points = [((715, 360),0),((348,414),90),((127,350),95),((581,432),270),((882,71),20),((970,278),0)]
        my_rand_points = [((755, 400),0)]
        (x,y),angle = random.choice(my_rand_points)
        self.car.center = (x,y)
        self.car.angle = angle
        self.car.velocity = Vector(4, 0)


    def update(self, dt):

        global brain
        global reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global orientation


        global obs


        # NEW GLOBALS
        global replay_buffer
        global seed
        global start_timesteps
        global eval_freq
        #global max_timesteps
        global save_models
        global expl_noise
        global batch_size
        global discount
        global tau
        global policy_noise
        global noise_clip
        global policy_freq
        global done
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global episode_reward
        global reward_window

        global episode_timesteps
        global main_img
        global image_size
        global reached_dest
        global last_time_steps
        # NEW GLOBALS


        longueur = self.width
        largeur = self.height
        if first_update:
            init()







        #if total_timesteps < max_timesteps:
        if True :

          # If the episode is done
          if done:


            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
              print("Total Timesteps: {} Episode Num: {} Timesteps diff: {} Reward: {} score: {}".format(total_timesteps, episode_num, total_timesteps - last_time_steps,episode_reward, episode_reward/(total_timesteps - last_time_steps)))
              brain.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
              last_time_steps = total_timesteps



            # if the agent have not reached destination and collided to walls then reinitialise the agent at a given point
            if  not  reached_dest:
                # initialize the car at new point
                self.serve_car()
                #cnn state calculation
                _,_,obs = get_target_image(main_img, self.car.angle, [self.car.x, self.car.y], image_size)
                save_cropped_image(obs, self.car.x, self.car.y, name = "initial")

                xx = goal_x - self.car.x
                yy = goal_y - self.car.y
                orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
                orientation = [orientation, -orientation]
                #cnn state calculation


            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            reached_dest = 0

          # Before start_timesteps , we play random actions
          if total_timesteps < start_timesteps:
            action = [random.uniform(-max_action_agent * 1.0, max_action_agent * 1.0)]
            #action = env.action_space.sample()
          else: # After start_timesteps, we switch to the model
            action = brain.select_action(np.array(obs), np.array(orientation))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
              action = (action + np.random.normal(0, expl_noise, size=action_len)).clip(-1*max_action_agent,max_action_agent)

          # The agent performs the action in the environment, then reaches the next state and receives the reward


          # ENV STEP PERFORM START
          if type(action) != type([]):
              self.car.move(action.tolist()[0])
          else:
              self.car.move(action[0])
          distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

          if sand[int(self.car.x),int(self.car.y)] > 0:
              self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)

              reward = -1
              if distance < last_distance:
                reward = -1.2
          else: # otherwise
              self.car.velocity = Vector(2, 0).rotate(self.car.angle)
              reward = -0.2
              if distance < last_distance:
                  reward = 0.1
              # else:
              #     last_reward = last_reward +(-0.2)



          if self.car.x < 40:
              reward_window.append(-3)
              self.car.x = 40
              reward = -1
#          elif self.car.x < 10:
#              reward = -1# * (10-self.car.x)

          if self.car.x > self.width - 40:
              reward_window.append(-3)
              self.car.x = self.width - 40
              reward = -1
#          elif self.car.x > self.width - 10:
#              reward = -0.1 * (self.car.x- self.width +10)

          if self.car.y < 40:
              reward_window.append(-3)
              self.car.y = 40
              reward = -1
#          elif self.car.y < 10:
#              reward = -1# * (10-self.car.y)

          if self.car.y > self.height - 40:
              reward_window.append(-3)
              self.car.y = self.height - 40
              reward = -1
#          elif self.car.y > self.height - 10:
#              reward = -1# * (self.car.y- self.height+10)

          if distance < 25:
              done = True
              reached_dest = 1
              if swap == 1:
                  goal_x = 1460
                  goal_y = 662
                  swap = 0
              else:
                  goal_x = 49
                  goal_y = 125
                  swap = 1
              reward = 2
          last_distance = distance

          # cnn state calculation
          _,_,new_obs = get_target_image(main_img, self.car.angle, [self.car.x, self.car.y], image_size)
          xx = goal_x - self.car.x
          yy = goal_y - self.car.y
          new_orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
          new_orientation = [new_orientation, -new_orientation]
          save_cropped_image(new_obs, self.car.x, self.car.y, name = "")
          # cnn state calculation

          reward_window.append(reward)

          if sum(reward_window[len(reward_window)-100:]) <= -188 or episode_timesteps % 1200 == 0 and episode_timesteps != 0:
              done = True
              reward_window = []


          # ENV STEP PERFORM END

          # We increase the total reward
          episode_reward += reward

          # We store the new transition into the Experience Replay memory (ReplayBuffer)
          replay_buffer.add((obs, orientation, new_obs, new_orientation, action, reward, done))

          # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
          obs = new_obs
          orientation = new_orientation
          episode_timesteps += 1
          total_timesteps += 1
          timesteps_since_eval += 1





# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1


            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear', size = (30,30))
        savebtn = Button(text = 'save', pos = (parent.width, 0), size = (30,30))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0), size = (30,30))
        addbtn = Button(text = "add", pos = (3*parent.width,0), size = (30,30))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        addbtn.bind(on_release = self.addroad)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(addbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

    # this button adds roads to the environment
    def addroad(self,obj):
        print("adding roads to the environment...")
        global im
        global main_img
        global sand
        # load masks with roads
        im = CoreImage("./images/MASK1.png")
        main_img = cv2.imread('./images/mask.png',0)
        sand = np.zeros((longueur,largeur))
        img = PILImage.open("./images/mask.png").convert('L')
        sand = np.asarray(img)/255
        print("sucessfully added  roads")

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
