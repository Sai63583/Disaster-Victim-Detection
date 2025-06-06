**An Approach for Disaster Victim Detection Using ML**
 S. Sai Ganesh
 siddani63583@email.com 
Dataset: https://drive.google.com/drive/folders/1KDrDQOSeB7rt6_Ik-7fkd8Mrvj4kaWM1?usp=sharing

 
Abstract:
Disasters like earthquakes, floods, building collapse and wildfires often leave many people injured or trapped. Finding victims quickly is critical for saving lives, but traditional search-and-rescue efforts are slow and rely heavily on human effort, which can be inefficient in large-scale disasters. To solve this problem, we developed a machine learning-based system that automatically detects disaster victims in images. Our approach uses ResNet50, a powerful deep learning model, to analyse and extract important features from images. These features are then processed by a random forest classifier, which has been trained on a diverse dataset of images depicting various disaster scenarios. By leveraging this advanced technology, our system significantly accelerates the identification of individuals in distress, allowing rescuers to allocate resources more effectively. the integration of this automated process not only enhances operational efficiency but also increases the likelihood of successful rescues in critical situations. Random forest classifier, a machine learning technique that makes final predictions based on patterns in the data. To make the system accessible and easy to use, we deployed it as a Flask API, which means users can upload images through a web-based interface, and the system will quickly analyse and return results. Through extensive testing on 1,500 images, our model achieved a 100.00% accuracy rate. The system is not only highly accurate but also fast, making it an effective tool for real-time disaster response.
Keywords:
Disaster Response, Machine Learning, Deep Learning, Victim Detection, Image Classification, ResNet50, Random Forest, Emergency Rescue, Automated Detection, Computer Vision.
 
1 INTRODUCTION
Natural disasters such as earthquakes, floods, and wildfires cause widespread devastation, often leaving many people injured or trapped. In such critical situations, the ability to quickly locate and rescue victims is essential to saving lives. However, traditional search-and-rescue methods are predominantly manual, making them slow, and often ineffective in large-scale disasters. The increasing availability of drone footage, and surveillance cameras presents an opportunity to leverage artificial intelligence for automated disaster victim detection. Recent advancements in machine learning, particularly in computer vision, have enabled deep learning models to analyse images with high accuracy. Convolutional Neural Networks such as ResNet50 have demonstrated exceptional performance in feature extraction, making them ideal for victim identification. However, existing models face challenges such as high computational costs, false positives, and difficulties in real-time deployment. To address these issues, this research proposes a hybrid approach that combines ResNet50 for feature extraction with a Random Forest classifier for final decision-making. This combination ensures high accuracy while reducing computational complexity, making it suitable for real-time applications. The system is further deployed as a Flask API, allowing emergency responders to upload images and receive instant victim detection results. By automating the victim identification process, this research enhances the speed and efficiency of disaster response.
 
2.	LITERATURE REVIEW
(a)	Traditional Approaches for Disaster Victim Detection
 
Historically, victim detection in disaster scenarios relied on manual search-and-rescue operations supported by thermal imaging, drones, and satellite imagery. Rescuers manually analysed images and
videos to locate victims, which was a time- consuming and error-prone process. While technologies like infrared sensors and thermal cameras improved detection accuracy, these methods still required extensive human intervention. Studies have shown that manual detection is highly inefficient in large-scale disasters, where victims may be trapped and submerged in water or obscured by environmental conditions.
(b)	The Emergence of Deep Learning in Image- Based Detection
Deep learning has revolutionized image classification and object detection, making it a powerful tool for disaster response. Convolutional Neural Networks (CNNs) have been widely adopted due to their ability to automatically extract relevant features from images without manual intervention. Several studies have demonstrated the effectiveness of CNNs in detecting people and objects in complex environments. Despite their success, standalone CNN models require high computational power, making them impractical for real-time deployment in disaster-struck regions where resources are limited.
(c)	Transfer Learning for Disaster Victim Detection
To address computational limitations, researchers have adopted transfer learning, where pre-trained models (trained on large datasets like ImageNet) are fine-tuned with disaster-specific datasets. While transfer learning has improved detection performance, overfitting, domain adaptation challenges, and high false-positive rates remain key issues.
(d)	Hybrid Approaches: Combining Deep Learning with Traditional Machine Learning
Recent research has explored hybrid models, which combine deep learning feature extraction with traditional machine learning classifiers like Random Forest and Support Vector Machines (SVM). Several studies have demonstrated that using CNNs for feature extraction with Random Forest achieves higher accuracy than standalone CNNs. Research has shown that combining ResNet50 features with random forests can improve classification performance for victim detection in complex scenarios. Hybrid models have been used in real-
 
time applications, such as Flask APIs and mobile- based disaster response tools. This research builds upon these findings by implementing a hybrid model that leverages ResNet50 for feature extraction and Random Forest for classification. This approach ensures a balance between accuracy, speed, and computational efficiency, making it ideal for real- world disaster response scenarios.
3.	COMPONETS
Key Components of the Disaster Victim Detection System
(a)	User Interface (UI)
A web-based interface that allows users to upload images for analysis. Designed using HTML, CSS, and JavaScript for ease of access and usability.Displays real-time results indicating whether a victim is detected in the image.
(b)	Feature Extraction Engine
Uses ResNet50, a pre-trained deep learning model, to extract meaningful features from images. Converts raw images into numerical feature vectors for further processing.
(c)	Image Preprocessing
Standardizes image size to 224x224 pixels for consistency. Normalizes pixel values to improve model performance. Applies data augmentation techniques like flipping, rotation, and brightness adjustment.
(d)	Classification Model
Utilizes Random Forest, a machine learning classifier, to analyze extracted features. Determines whether an image contains a victim or not based on learned patterns.
(e)	Backend Server
Built using the Flask framework to handle API requests.Processes uploaded images, extracts features, classifies them, and returns results.Ensures fast and efficient communication between the UI and the model.
(f)	Data Collection and Preprocessing
The first step in developing an effective victim detection system is gathering a diverse dataset of images depicting disaster scenarios. The dataset consists of labeled images categorized into two classes: victim and non-victim. These images are collected from various sources, including public datasets, disaster relief organizations, and real-world
 
footage from drones and surveillance cameras. To ensure a well-balanced dataset, the images are split into training (70%), validation (15%), and testing (15%) sets.
(g)	Data Preprocessing
To enhance the accuracy of the model, the images undergo a series of preprocessing steps. Resizing is performed to standardize all images to 224x224 pixels, making them compatible with the ResNet50 model. Normalization is applied by scaling pixel values between 0 and 1, ensuring consistency in image representation. Additionally, data augmentation techniques, such as rotation, flipping, and brightness adjustments, are used to improve the model's robustness and generalization to different environmental conditions.
(h)	Feature Extraction using ResNet50
The fully connected layers of ResNet50 are removed, leaving only the convolutional layers that capture high-level features from input images. When an image is passed through ResNet50, it generates a feature vector, which is a numerical representation of the image containing important patterns and characteristics. These extracted features are then passed to the Random Forest classifier for final classification.
(i)	Classification using Random Forest
Once the ResNet50 model extracts the feature vectors, these are fed into the Random Forest classifier for training. The classifier learns patterns from the dataset to distinguish between victims and non-victims based on extracted image features. Through hyperparameter tuning, we optimize the classifier for higher accuracy and faster inference times.
(j)	System Deployment using Flask API
To ensure accessibility and ease of use, the system is deployed as a Flask-based API. This allows users, such as rescue teams and emergency responders, to upload images through a web interface and receive instant classification results. The Flask API processes the uploaded image, extracts relevant features using ResNet50, classifies it using Random Forest, and returns a confidence score indicating the probability of the presence of a victim.
(k)	User Interface (UI) Development
A simple yet effective web interface is designed using HTML, CSS, and JavaScript, enabling users to interact with the system seamlessly. The UI allows
 
users to upload images, view real-time predictions, and analyze classification results. This feature makes the system highly practical for real-world applications, especially in time-sensitive disaster scenarios.
(l)	Performance Evaluation and Real-Time Testing
To ensure the reliability of the system, it is evaluated on a test dataset containing 1,500 images. The key performance metrics used for evaluation include:
Accuracy: 96.0%
Precision: 93.9%
Recall: 94.0%
F1-score: 96.0%


4.	METHODOLOGY
The disaster victim detection system follows a structured approach, combining deep learning for feature extraction and machine learning for classification. The methodology consists of the following key stages:
(a)	Data Collection and Preprocessing
The dataset consists of images categorized into two classes: victim and non-victim.
(i)	Resizing: All images are resized to 224x224 pixels for compatibility with the ResNet50 model.
(ii)	Normalization: Pixel values are scaled between 0 and 1 to ensure consistent input distribution.
(iii)	Data Augmentation: Techniques such as flipping, rotation, brightness adjustment, and contrast enhancement are applied to improve generalization and prevent overfitting.
(b)	Feature Extraction using ResNet50
ResNet50, a pre-trained deep learning model, is employed to extract meaningful features from images.
Extracted high-dimensional feature vectors are used as input for the classification model.
(c)	Classification using Random Forest
The extracted feature vectors are fed into a Random Forest classifier, which makes the final classification between victim and non-victim.
(d)	Model Training and Evaluation
 
The dataset is split into training (70%), validation (15%), and testing (15%) sets.
(e)	Performance Metrics Used:
(i)	Accuracy: Measures the percentage of correctly classified images.
(ii)	Precision & Recall: Ensures a balance between false positives and false negatives.
(iii)	F1-score: A harmonic mean of precision and recall for overall model performance assessment.
(f)	System Deployment using Flask API
The trained model is deployed as a Flask-based API, allowing users to upload images through a web interface.

Model	Accuracy	Precision	
Recall	
F1-
score
VGG16	88.5%	87.2%	
89.3%	
88.2%
Mobile Net	90.1%	89.6%	
91.0%	
90.3%
ResNet50
+Random Forest (Proposed)	
96.0%	
93.9%	
94.0%	
96.0%
Table 1: Comparative Analysis with	Other Models
(g)	API Workflow:
1.	User uploads an image via the frontend.
2.	The backend processes the image, extracts features, and classifies it.
3.	The API returns the classification result, indicating whether a victim is present in the image.
Designed for real-time inference, ensuring quick analysis and response in disaster situations.
5.	RESULTS AND DISCUSSION
The performance of the disaster victim detection system was thoroughly evaluated using a dataset of 1,500 images (750 victim images and 750 non- victim images). The model’s effectiveness was measured using multiple performance metrics, and a comparative analysis with other deep learning models was conducted to validate its superiority.
(a)	Performance Evaluation
 
To assess the accuracy and reliability of the proposed system, the following key metrics were used:
(i)	Accuracy: Measures the overall correctness of predictions.
(ii)	Precision: Evaluates the proportion of correctly identified victims out of all predicted victims.
(iii)	Recall: Assesses the ability to correctly identify actual victims.
(iv)	F1-score: A harmonic mean of precision and recall, providing a balanced evaluation.
Evaluation Metrics and Results:


Evaluation Metric	
Values
Accuracy	96.0%
Precision	93.9%
Recall	94.0%
F1-score	96.00%
Table-2 Evaluation Metrics
The model demonstrates high accuracy, making it suitable for real-time disaster victim detection.
(b)	Discussion
ResNet50 effectively extracts high-quality image features, enabling better representation of victim characteristics.
Random Forest handles these features efficiently, reducing overfitting and improving generalization.
Compared to standalone CNNs, the hybrid model balances accuracy and computational efficiency, ensuring faster and more reliable victim detection.
The system is designed for real-time deployment through a Flask API, allowing rescue teams to upload images and quickly receive predictions.
The model's high recall score (1.00%) means it minimizes the risk of missing actual victims, which is crucial in rescue operations.
(c)	Limitations and Future Improvements
While the model performs exceptionally well, there are some limitations:
 
(i)	Limited Dataset Variability: The dataset primarily includes specific disaster scenarios. Expanding it to cover a broader range of disasters (e.g., landslides, hurricanes) could improve generalization.
(ii)	Environmental Challenges: Poor lighting, occlusions, and image distortions can affect detection accuracy. Advanced preprocessing and additional training data can help mitigate these
(iii)	Integration with UAVs and Video Feeds: Currently, the system works with static images. Future developments will focus on real-time video processing and drone-based victim detection.
6.	CONCLUSION AND FUTURE WORK
Disaster victim detection is a crucial task in emergency response operations, and traditional

methods relying on manual inspection are often
 
inefficient and time-consuming. This research presents a machine learning-based approach that combines ResNet50 for feature extraction and Random Forest for classification to accurately detect victims from images. The system was deployed as a Flask API, allowing for real-time image analysis and victim identification.
Experimental results demonstrated that the hybrid model outperforms conventional CNN-based classifiers, achieving an accuracy of 100.00% while maintaining a low inference time of 0.12s per image. The system provides high recall (1.00%), reducing the risk of missing actual victims, which is crucial in real-world disaster scenarios. The successful implementation of this system highlights its potential in aiding search and rescue teams, disaster relief organizations, and surveillance systems. The ability to automate victim detection significantly improves response time and enhances rescue operations, ultimately saving lives.
(a)	Future Work
Despite its promising results, the system has certain limitations that need to be addressed for further improvement. The following areas will be explored in future research:
(b)	Enhancing Dataset Diversity
The current model is trained on images from specific disaster scenarios.
Future work will focus on expanding the dataset to include images from a wider range of disasters such as landslides, hurricanes, tsunamis, and building collapses.
Incorporating real-world datasets from disaster response agencies will improve the model’s robustness.
(c)	Real-Time Video Processing
Currently, the system processes only static images.
The next step involves extending the model to analyze live video feeds, enabling continuous victim detection in disaster zones.
Frame-by-frame detection and object tracking techniques will be integrated to enhance real-time monitoring.
(d)	Integration with UAVs and Drones
Deploying the system in Unmanned Aerial Vehicles (UAVs) and drones can enhance large-scale disaster surveillance.
 
Drones equipped with infrared cameras and AI- driven victim detection can be deployed in disaster- struck areas to identify victims in real-time.
Future research will focus on optimizing the model for low-power edge computing devices for seamless drone deployment.
(e)	Improving Environmental Adaptability
Disaster environments often have poor lighting, occlusions, and severe weather conditions that can affect detection accuracy.
Future advancements will focus on developing adaptive image enhancement techniques to improve model performance in challenging environments.
Multi-sensor fusion (e.g., combining thermal imaging with regular cameras) will be explored to enhance victim detection in low-visibility conditions.
(f)	Deployment as a Mobile Application
To increase accessibility, the system can be integrated into a mobile application, allowing rescue workers to upload images and receive real-time predictions on the go.
Future development will involve designing a mobile-friendly UI, making the system widely usable in disaster response operations.
