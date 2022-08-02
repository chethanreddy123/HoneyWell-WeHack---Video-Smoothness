# HoneyWell-WeHack---Video-Smoothness

## What is the problem?
### During video acquisition, due to sensor malfunction, poor network bandwidth, acquired videos suffers from blurry artifacts, different noise effects. 
A surveillance camera developer needs to test the product by manually verifying these videos void from any unwanted artifacts: <br>
➔ Time Consuming<br>
Videos recorded for days<br>
➔ Cost-ineffective<br>
Takes labour to manually go through<br>
➔ Inaccurate<br>
Human can make mistakes going
through hours of videos

### Solution:

WeDio can give you accurate timestamps of 5 major kinds of disruptions in your video :<br>
1. Lagging<br>
● Calclulating the current fps, for every 3s<br>
● Making list of change of fps with
previous fps,<br>
● Putting the threshold to 10 <br>
● If change in fps is >10, it is a lag <br>

2. Blurriness<br>
● Variants of laplacian, calculating the variance<br>
● Putting threshold to 100,<br>
● If <100 then it is blurred<br>

3.Color Variation<br>
● We are getting the current color of the pixels<br>
● Using a norm score to verify its black<br>

4.Quality Checking<br>
● Brisque algo<br>

5.Freezing<br>
● Getting the difference of the
pixels of the two frames with a
delay of 3s

###  Brisque algo
![MicrosoftTeams-image](https://user-images.githubusercontent.com/69640722/182423753-ac413e0d-0a9a-4799-bfc3-31e3e161c3ca.png)

### Video Explanation:

https://user-images.githubusercontent.com/69640722/182424039-de3cd6ac-15fd-4537-a901-434f37d3ec13.mp4





![adf](https://user-images.githubusercontent.com/69640722/182422760-af5a504f-2ef0-40d8-86cc-8107495bdb18.jpg)
