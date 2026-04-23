Project goal, at first, figure out how to work Siamese trackers in general and Nanotrack especially + Kalman filter.
Almost all code was taken from https://github.com/HonglinChu/SiamTrackers, but I figured out how it works, figured out what kind of Siamise tracker I need and added some improvements.

Improvements:
- Kalman filter in nano_tracker.py: At first - stabilization, the most common usage for tracking. At second - lost mode, when tracker is lost it will be conducted only by Kalman filter trajectory.
- Re-detection in nano_tracker.py (_redetect): If tracker is lost - it will try to use search with doubled radius to find original object. 

So, the combination of these 2 improvements will provide mechanism (mostly based on assumptions) that help to re-detect objects after disappearing it behind the other objects.
Obviously, it's need to be improved in different ways (speed (rewrite to c++, use onnx or openvino), more robust re-detection, maybe some kalman filter improvements, more precisely figure out when we have to turn on lost mode), it would be improved, I tested big amount ways to improve this tracker (can be viewed by commit history + I actually started in the different repo) and will test more, but for now, I don't have time :)

Pain: As far I didn't understand how to improve original model. I tried to train pre-trained model, on GOT-10k dataset but it deacresed AO on GOT-10k test data, so as far I skip this part (actually I think that it has no sense for this case, but I can be wrong (can u explain plz?)) 

Re-detection example, here I skip 100 frames of video, so tracker is lost in the next frame:

<img width="450" height="250" alt="download (1)" src="https://github.com/user-attachments/assets/19aa6de3-44e8-4891-ab84-30da13b7f9e6" />

Here is re-detection after losting (and leading only by Kalman trajectory):

<img width="333" height="344" alt="Untitled design (1) (1) (1) (1)" src="https://github.com/user-attachments/assets/7fc2b537-f5a9-4252-82e3-d475920c5840" />
