# Master-Thesis
A recommender system provides personalized recommendations to the user in a large space of possible options. Most of the existing recommender systems utilize user’s profile data for recommending an object. However, these conventional recommender systems are under scrutiny due to strict personal data protection laws around the world. In this thesis, we present a graph-based scalable and novel approach for the recommendation which doesn’t depend on the profile data for predictions. We evaluate our method against a non-machine learning-based approach on an extensive transaction (700k transactions with 150k different items) dataset from the retail domain. We iteratively improved the performance of our approach. The proposed method first uses a close to linear runtime knowledge graph embedding algorithm, Pyke. The initial predictions from this approach were low. The reason behind the low prediction was investigated and rectified by replacing Pyke with a convolutional complex knowledge graph embedding algorithm, Conex. 
Please look at the wiki page for detailed description