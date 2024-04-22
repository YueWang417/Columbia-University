K-means Parameters:

We stop if max iterations reach 50 or if it achieves an accuracy of 0.7 to find a balance between running time and missing better clusters. Because we don't want the algorithm to run for too long and waste time, but we also want it to be thorough in identifying the stop sign. 

We set k = 11, here k is the number of color groups we want the algorithm to find, as we believe that's roughly the number of main colors present in the image.

We select KMEANS_PP_CENTERS to initialize cluster centers in a way to speeds up convergence.


Challenges for the K-means Approach:

If the image is too bright, the stop sign's distinct red may fade. This makes it tough for the algorithm to pick out the stop sign specifically. In addition, If there are several red objects near the stop sign, the algorithm might mistakenly consider them as part of the sign.