

[![PyPI Package Link](https://badge.fury.io/py/symbtools.svg "PyPI Package Link")](https://badge.fury.io/py/symbtools) [![Travis CI Unit Test Result badge](https://travis-ci.org/TUD-RST/symbtools.svg?branch=master "Travis CI Unit Test Result badge")](https://travis-ci.org/TUD-RST/symbtools) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.275073.svg)](https://doi.org/10.5281/zenodo.275073)


(English version below)

Allgemeines
===========
Das Paket `symbtools` enthält eine Sammlung von Funktionen für
symbolischen Rechnungen, die bei der Untersuchung nichtlineare dynamischer
Systeme im Rahmen der Regelungs- und Steuerungstheorie auftreten. Speziell ist im Modul `modeltools` Funktionalität gebündelt, die zur Herleitung, Analyse und Simulation von Modellgleichungen mit Hilfe der Lagrange-Gleichungen 1. bzw. 2. Art dient. Dieses Modul unterliegt aktuell (August 2019) einer Restrukturierung. Eine separate Dokumentation ist in Vorbereitung. Vorerst wird auf die [unittests](https://github.com/TUD-RST/symbtools/blob/master/symbtools/test/test_modeltools.py) und auf das Repositorium <https://github.com/cknoll/beispiele> verwiesen.

Der Programmcode hat den Status von "Forschungscode",
d.h. das Paket befindet sich im Entwicklungszustand.
Trotz, dass wesentliche Teile durch Unittests abgedeckt sind, enthält der Code
mit einer gewissen Wahrscheinlichkeit Fehler.



General Information
===================
The package symbtools contains collection of functions for symbolic
calculations, which occur along with the investigation of nonlinear
dynamical systems in the field of control theory.

Note that, the package is in development state. Despite that a substatitial
part is covered by unittest, the code will contain bugs with some probability.


Installation
============
Make sure you have the following dependencies installed:

- sympy
- numpy
- scipy
- ipython

Get symbtools using PyPI::

    $ pip install symbtools

or the latest git version::

    $ git clone https://github.com/TUD-RST/symbtools.git

