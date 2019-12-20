## Attention visualization for Deep Learning models in **PyTorch**

This repository contains some features visualization methods for DL models in **PyTorch**
Another repo for more techniques: [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

> codes/ is the folder of source scripts

> data/ is the folder of some samples

> model/ is the pretrained ResNet34 model on ImageNet

> results/ is the folder for attention / saliency / features maps


#### (CAM) Class Activation Map


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="30%" align="center"> Original Images </td>
			<td width="30%" align="center"> Attention Maps </td>
			<td width="30%" align="center"> Overlapped Images </td>
		</tr>
		<tr>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/data/plane.jpeg"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/plane_404_CAM.png"> </td>
			<td width="30%" align="center"> <img src="https://github.com/gatsby2016/FeatsVisDL/blob/master/results/plane_404_overlap.png"> </td>
		</tr>
	</tbody>
</table>



#### Layers feature maps visualization














### Note for Me.
```
git init
git add README.md
git commit -m "first commit"
git remote add origin git@github.com:gatsby2016/FeatsVisDL.git
git push -u origin master
```
