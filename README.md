## Maps visualization for Deep Learning models in **PyTorch**

This repository contains some features visualization methods for DL models in **PyTorch**.

> `codes/` is the folder of source scripts

> `data/` is the folder of some samples

> `model/` is the pretrained ResNet34 model on ImageNet

> `results/` is the folder for **attention / saliency / features maps**

Another repo for more techniques: [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)


### CAM (Class Activation Map)

**Paper reference:** [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)

To visualize the model where it focus on by activation maps.

The limitation is that the model must has a **Global Pooling** followed by one **fully connected layer** to classes.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="30%" align="center"> Original Images </td>
			<td width="30%" align="center"> Activation Maps </td>
			<td width="30%" align="center"> Overlapped Images </td>
		</tr>
		<tr>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/data/plane.jpeg"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/CAM/plane_404_CAM.png"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/CAM/plane_404_overlap.png"> </td>
		</tr>
		<tr>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/data/rugbyball.jpg"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/CAM/rugbyball_768_CAM.png"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/CAM/rugbyball_768_overlap.png"> </td>
		</tr>
	</tbody>
</table>


### Feature maps visualization on Layers

To visualize the **features maps** after each layer, which can also be viewed as the method for **DL features extraction**.

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="30%" align="center"> Original Image </td>
			<td width="30%" align="center"> Maps after 1st maxpool </td>
			<td width="30%" align="center"> Maps after Layer1</td>
		</tr>
		<tr>
			<td width="15%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/data/ball.jpg"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/featsmapVis/ball/gif/maxpool.gif"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/featsmapVis/ball/gif/layer1.gif"> </td>
		</tr>
    <tr>		<td width="30%" align="center"> Maps after Layer2 </td>
			<td width="30%" align="center"> Maps after Layer3 </td>
			<td width="30%" align="center"> Maps after Layer4 </td>
		</tr>
		<tr>
			<td width="60%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/featsmapVis/ball/gif/layer2.gif"> </td>
			<td width="120%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/featsmapVis/ball/gif/layer3.gif"> </td>
			<td width="240%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/featsmapVis/ball/gif/layer4.gif"> </td>
		</tr>
	</tbody>
</table>





### Note for myself.
```
git init
git add README.md
git commit -m "first commit"
git remote add origin git@github.com:gatsby2016/FeatsVisDL.git
git push -u origin master
```
