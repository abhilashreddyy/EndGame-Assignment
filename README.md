# EndGame-Assignment

## Implementation of self driving car using TD3 algorithm

# Training self driving car on TD3 deep reinforcement learning algorithm 

## Summary of what is done 
- Added CNN to calculate state from image
    - replaced the sensor input with cropped & rotated Image input
- Inputting the orientation of the car to neural network
    - __NOTE__ : Please refer to Actor and Critic Images to understand better
    - Without orientation car was stumbling here and therewhile staying on road. 
    - So added orientation to acknowledge the agent to reach destiny
- Shifted the entire update operation to car.py from brain.update()

## Detailed Summary :
- Started solving the assignment by fixing the TD3 into asssignment 7 
    - Initially, Implemented the update function in the ai.py module
    - Then i messed up some part in collecting the last state and rewards (___Realising this had consumed a lot of time___). I tried to restructure the entire update function and write my custom update function in TD3 class from what i have understood. But I dont know for some reason it was not working quiet well. Car was always rotating around itself
    - After that, restructured the entire code again and shifted the update function into car.py update function
- Then added done condition to the code. 
- Now the car was able to go straight on road but was unable to take turns.
    - To solve this i have tried to increase exploration coefficient(__expl_noise__) and changed the different shapes of output channels in CNN layer. This had resulted in some exploration of car .But notthing significant
- Then i have tried to play with the rewards
    - Incresased the reward for living and going away from targetfrom -0.2 to 0.2
    - This resulted in car trying to stay on road but unable to learn to reach the destination
    - Later decreased the reward of moving towards the destination and staying on road. This resulted in car tending more to travel on sand. 
- Added orientation to the network by appending it to the output of the CNN layer. I havent quiet tested it thoroughly

Most of my time was consumed in merging the TD3 into assignment7. I did not get much time play with different hyper parameters or Think more about what more features can i add to actions and states parameters so that model can train better. 

I would like to try following things if i got extension for assignment:
- Right now done condition is set to true if the car travels 10-20 steps on sand. I will change the done condition so that car can learn how to get back on road even it is lost
- Figure out the temperature parameter and its significance and try implement it in assignment 10
- Add the distance as a state parameter to the network
- Right now i am making the done = True after evern N number of steps. Remove that done condition and observe the results
- Rethink what all the parameter can used as actions , what all the parameters can be sent to state and restructure reward system for better performance
- Increase the number of initial steps and train
- Make padding At the border of the image

I believe that trying these things out will defenitely open gates for new ideas and intutions and Improve the model further



## Observations
- After training with this code car was able to understand how to keep itself on roads. But was unable to learn how to reach destiny. This is the major thing i need to focus on.





__Refer__ [this](https://youtu.be/2h8b4orhTT4) link to see some video of how car was training.
- These are some small instances of recording while the model was training.
- It can be clearly observed that the model is trying to stay on road while not very confident about reaching the destination

__NOTE__ : I have recorded the video before removing the sensor image (dots infront of car) .Apologies for that\
__NOTE__ : I still kept some of the old code commented without deleting. So that reader can find it easy to correlate the old code with the new one\
__NOTE__ : I dont have GPU so I was unable to do much hyper parameter tuning
 
 ### New Actor model
 ![actor](image_pres/final_actor.jpg)
 
 ### New critic model
 ![critic](image_pres/final_critic.jpg)
