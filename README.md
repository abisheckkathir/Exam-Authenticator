<h1 align="center">Exam Authenticator</h1>

## About the project
In the world of identity theft and ever evolving crimes, the issue of security is very paramount in any organization.

Biometrics is one such domain of Computer Science, which provides solution in terms of security through interpretation of human characteristics such as physical traits and behaviors to make it more reliable since these traits are unique for everyone. Hence, we intend to provide a secure validation through means of visual and behavioral biometrics such as facial recognition and signatures respectively.
The examination portal is used to authenticate students by using the following biometric traits:
1. Face - Physiological trait
2. Signature - Behavioural trait

![Flow](Flow.pmg)

* Facial recognition carries on 1:N identification where it compares the given template against all other templates already available in the database.
* Signature verification carries on 1:1 verification where it compares the given template by the user against the template of the given user with the help of the User ID.

<img src="https://i.pinimg.com/originals/aa/2f/f3/aa2ff34a12fde12ee717a4e8ebd6571d.jpg" width="500" align='center'>

The portal also provides the option to register. Upon registering, the user's templated will be successfully encoded and saved in the database.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

* [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads) - World’s most popular Python distribution platform

Please read [Installation Instructions](https://docs.anaconda.com/anaconda/install/) for details on how to install and setup Anaconda on your system.

### Installing

A step by step series of examples that tell you how to get a development env running

Create a new environment from the .yml file

```
conda env create -f environment.yml
```

And activate

```
conda activate bio
```

You are ready to run the Web App now...

## Running the App

Run this after following the installation steps:

```
flask run
```

## Built With

* [Flask](https://flask.palletsprojects.com/) - The web framework used
* [OpenCV](https://docs.opencv.org/master/) - Computer Vison Library used
* [face-recognition](https://pypi.org/project/face-recognition/) -Face Recognition Library used
* [TensorFlow](https://www.tensorflow.org/api_docs) - Machine Learning Library used

## Authors

* **Abisheck Kathirvel** - [abisheckkathir](https://github.com/abisheckkathir) - *Python and Integration*
* **Sivasini Netra** - [sivasini](https://github.com/sivasini) - *Flask and UI*

See also the list of [contributors](https://github.com/abisheckkathir/Exam-Authenticator/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Made this as a part of the Biometrics course in junior year

## References
* For face recognition - [https://www.mygreatlearning.com/blog/face-recognition/](https://www.mygreatlearning.com/blog/face-recognition/)
* For signature recognition -[https://github.com/Harshitb1/AxisBankAiChallenge](https://github.com/Harshitb1/AxisBankAiChallenge)
