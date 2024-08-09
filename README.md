
# Motion Prediction of Automated Vehicle

### Goal of Motion Prediction

It aims to forecast the upcoming headings, velocities, and positions of moving objects in the surrounding environment over a limited time. The expected trajectories also help us to ensure that the ego vehicle's planned path won't collide with any other objects in the future.

<img src="https://raw.githubusercontent.com/mangeshvir/Motion-Prediction-using-deep-learning/main/Basic%20Idea.png" width="450" height="350">

--> The goal of this project is to forecast the trajectory of an ego vehicle depending on the objects in its immediate environment such as cars, pedestrians, motorbikes or bicycles.

- **Libraries** - Pandas, PyTorch, PyTorch Lightening, Scikit learn.

- **Language** - Python

### Dataset used -  [inD Dataset](https://www.ind-dataset.com/)

 - The drone recorded the road users' naturalistic behavior at 4 different German intersections over a certain period of time. 

<img src="https://user-images.githubusercontent.com/108230926/218285508-f394bc36-7ba5-4984-bd42-182beac3e316.png" width="300" height="300">   <img src="https://user-images.githubusercontent.com/108230926/218285550-76b10c61-9a69-415d-ad73-2b64cc41867d.png" width="300" height="300">

<img src="https://user-images.githubusercontent.com/108230926/218285555-6e9dacf8-47c7-4f19-b208-c01d88fc1346.png" width="300" height="300"> <img src="https://user-images.githubusercontent.com/108230926/218285562-cb0655a1-1427-414f-ae16-a4de4fab5334.png" width="300" height="300">

 - 33 Recordings at four different recording locations

- One recording is ~ 15 minutes

 - The dataset included more than 13,500 objects such as vehicles, bikes, and pedestrians at junctions in the following [format](https://www.ind-dataset.com/format).



### Types of Motion Prediciton algorithm

<img width="721" alt="image" src="https://user-images.githubusercontent.com/108230926/218285576-05e452a5-8f6c-4eba-92f4-edbceee3a149.png">

For the project, the interaction aware motion model was used which consisted of neural network architecture.

<!-- This is a comment in a Markdown file 

### Neural Network-Based Motion Prediction

- **Data Normalization**
<img width="801" alt="image" src="https://user-images.githubusercontent.com/108230926/218285599-7c3caa2f-08d9-4820-b825-f0199a9f48dd.png">

- **Splitting Train-Test data**
<img width="666" alt="image" src="https://user-images.githubusercontent.com/108230926/218285628-bee3112d-948f-413b-b053-c9d24ea898e2.png">

- **Training the neural network model**
<img width="837" alt="image" src="https://user-images.githubusercontent.com/108230926/218285643-1abbed58-6615-4ecc-8703-9d42107b2a81.png">

- **Testing the model using test data**
<img width="844" alt="image" src="https://user-images.githubusercontent.com/108230926/218285694-4f0f8c4c-796a-4bb6-b596-5359f51219a3.png"> -->

### Methodology

<img width="846" alt="image" src="https://raw.githubusercontent.com/mangeshvir/Motion-Prediction-using-deep-learning/main/Methodology.png">


<!-- ### Our Neural Network Model Architecture
<img width="834" alt="image" src="https://user-images.githubusercontent.com/108230926/218285758-e97581c7-23b6-452b-8682-87788ea42d19.png"> -->

### Hyperparameters
<img width="844" alt="image" src="https://raw.githubusercontent.com/mangeshvir/Motion-Prediction-using-deep-learning/main/Hyperparameters.png">

### Result
<img src="https://user-images.githubusercontent.com/108230926/218286185-d04cb3f0-ca4c-4567-a026-ec01bc7cdeff.png" width="450" height="300">   

<img src="https://raw.githubusercontent.com/mangeshvir/Motion-Prediction-using-deep-learning/main/Results.png">

- The validation (test) dataset line is stable after 40 epochs
- The prediction of test dataset (unseen) will be close to accurate
  
--> Although the constant acceleration model exhibits minimal error, its application in real-world scenarios is deemed impractical due to oversimplified assumptions. 
In contrast, **Long Short-Term Memory (LSTM)** networks emerge as the most pragmatic choice for predictive modeling, given their capacity to incorporate and analyze intricate time-series data patterns.

--> The average displacement error (average Euclidean distance between the position prediction and ground truth) is found to be 0.9 m

--> The average absolute heading error (average absolute deviation of the heading of the ego vehicle from the ground truth value) is 3.62 degrees

### Future Scope

- Vehicular communication (V2V, V2X, V2I)
- Improved Prediction Accuracy
- Real-time Applications
- Behavior Prediction and Human Intent Recognition
- Can use Kalman Filter for motion prediction
