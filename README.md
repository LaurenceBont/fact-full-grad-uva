# Fairness, Accountability, Confidentiality and Transparency in AI
 This projects 

## People who worked on this project:
Daan Le - 11329866 - daan_le@hotmail.com\
Mathieu Bartels - 11329521 - mathieubartels@gmail.com\
David Wessels - 11323272 - davidwessels15@gmail.com\
Laurence Bont - 11198788 - laurencebont@gmail.com\
## Dependencies?
To get all dependencies for this paper use pipenv or conda

### pipenv:
`pip install pipenv`\
`pipenv shell`\
`pipenv sync`\

Now al packages needed for this project are installed!

### Data
For these experiments to run efficiently it is advised to download the pre-trained models and pre adjusted datasets.

unpack the datasets in the 'dataset' folder\
[pixel_perbutation](https://we.tl/t-vRK8oyPxoo)
[roar](https://we.tl/t-6EMIrE3Kct)
[extra-test](https://we.tl/t-9SXz30whky)

unpack the models in the saved-models folder
[models](https://we.tl/t-1n8BFJlouY)

### How to run?

Look at some examples in [the .ipynb](https://github.com/LaurenceBont/fact-full-grad-uva/blob/master/results.ipynb)
or run

`python main.py --experiment [roar|extra|pixel_perturbation]`

## Research
This project is based on a paper. Make sure to check out this paper, and the original github! 

[Full-Gradient Saliency Maps](https://github.com/idiap/fullgrad-saliency)

```
@inproceedings{srinivas2019fullgrad,
    title={Full-Gradient Representation for Neural Network Visualization},
    author={Srinivas, Suraj and Fleuret, François},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2019}
}
```
