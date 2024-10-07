<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">Benchmark Study on the Accuracy and Energy Consumption of TinyML Models</h3>

  <p align="center">
    A benchmark study on the accuracy and energy consumption of AlexNet, ResNet18, and VGG16 models reduced with model size minimization techniques
    <br />
    <a href="https://github.com/antgioia/benchmarkTinyML"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project
It is a project that allows the creation of minimized models using techniques such as unstructured random pruning, unstructured global pruning, unstructured pruning, channel-wise structured pruning, dynamic quantization, and low-rank approximation. Furthermore, once the models are obtained, it is possible to conduct a benchmark study to visualize the accuracy and sustainability metrics of the models both with and without minimization.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This section lists the main technologies used to develop:

- [![Python][python-shield]][python-url]
- [![Pandas][pandas-shield]][pandas-url]
- [![Pytorch][pytorch-shield]][pytorch-url]
- [![Scikit-learn][scikit-learn-shield]][scikit-learn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Here is an example of how you can set up the project locally.
To get a local copy up and running, follow these simple steps.

### Prerequisites

There are no specific prerequisites

### Installation

1. Clone the Repository:
   ```sh
   git clone https://github.com/antgioia/benchmarkTinyML.git
   ```
2. Install Dependencies:
   ```sh
   cd ml
   pip install -r requirements.txt
   ```
### Usage

1. To create minimized models, run:
   ```sh
   python createMinimizedModels.py
   ```
2. To perform benchmark studies for accuracy, run:
   ```sh
   python benchmark.py
   ```
3. For o conduct sustainability benchmark studies, run:
   ```sh
   python emissions.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what makes the open source community such an amazing place to learn, inspire and create. Any contributions you make will be **very much appreciated**.

If you have a suggestion that could improve this project, please fork the repository and create a pull request. You can also simply open an issue with the ‘enhancement’ tag.
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m ‘Add some AmazingFeature’`)
4. Push on the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Domenico Antonio Gioia - [LinkedIn](https://www.linkedin.com/in/domenico-antonio-gioia-42541a1a2/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/antgioia/benchmarkTinyML.svg?style=for-the-badge
[contributors-url]: https://github.com/antgioia/benchmarkTinyML/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/antgioia/benchmarkTinyML.svg?style=for-the-badge
[forks-url]: https://github.com/antgioia/benchmarkTinyML/network/members
[stars-shield]: https://img.shields.io/github/stars/antgioia/benchmarkTinyML.svg?style=for-the-badge
[stars-url]: https://github.com/antgioia/benchmarkTinyML/stargazers
[issues-shield]: https://img.shields.io/github/issues/antgioia/benchmarkTinyML.svg?style=for-the-badge
[issues-url]: https://github.com/antgioia/benchmarkTinyML/issues
[license-shield]: https://img.shields.io/github/license/antgioia/benchmarkTinyML.svg?style=for-the-badge
[license-url]: https://github.com/antgioia/benchmarkTinyML/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/domenico-antonio-gioia-42541a1a2/
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[pandas-shield]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/