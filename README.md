**Robust and Effective Watermarking for GPS Trajectories**  
 

Four different implementations of the watermarking algorithm are present in the folder 'Code': 
1. W-Trace 
2. TrajGuard 
3. IMF
4. SVD

Each method consists of different Python files for :  
a) Watermark insertion- watermarkInsertion.py  
b) Adding noise to the trajectory data: AddotherNoises.py, AddhybridAttack.py, AddinterplotationAttackFFT.py,  AdddoublEmbeddingAttack.py, AddinterplotationAttackFFT.py  
c) Extracting Noise from Noise data: extractWatermarkFromNoises.py, extractwatermarkFromInterpolate.py  
d) For utility applications, refer to map matching code at Code/W-Trace/VallahMapMatching.ipynb and <a href="https://github.com/sobhan-moosavi/DCRNN"> driver identification code </a> 
   
 **Tools:-**  
   * Python(3.7) 

![alt text](https://github.com/Rajjat/watermarkingTrajectory/blob/master/watermark_img.png)  
Here, we show the visualization of original, watermarked, and noised trajectories.  
Green line: Original Trajectory  
Purple line: Watermarked Trajectory  
Yellow line: Noised Trajectory  
