# EPEngine

[![Release](https://img.shields.io/github/v/release/szvsw/epengine)](https://img.shields.io/github/v/release/szvsw/epengine)
[![Build status](https://img.shields.io/github/actions/workflow/status/szvsw/epengine/main.yml?branch=main)](https://github.com/szvsw/epengine/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/szvsw/epengine/branch/main/graph/badge.svg)](https://codecov.io/gh/szvsw/epengine)
[![Commit activity](https://img.shields.io/github/commit-activity/m/szvsw/epengine)](https://img.shields.io/github/commit-activity/m/szvsw/epengine)
[![License](https://img.shields.io/github/license/szvsw/epengine)](https://img.shields.io/github/license/szvsw/epengine)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14814542.svg)](https://doi.org/10.5281/zenodo.14814542)

This is a repository for managing distributed queues of simulations with Hatchet.

- **Github repository**: <https://github.com/szvsw/epengine/>
- **Documentation** <https://szvsw.github.io/epengine/>

The main goal is to empower large scale distributed EnergyPlus simulations, however the patterns herein are abstracted enough to be easily adapted to other engineering disciplines and projects with embarrassingly parallel workloads.

The main methodology is to define leaf workflows representing one independent unit of simulation work which can be allocated in a scatter-gather pattern or in a recursively subdivided scatter-gather pattern.

Currently, it supports working with arbitrary pre-generated EnergyPlus IDF files, or with dynamically runtime-generated simulations built with [EPInterface](https://github.com/szvsw/epinterface), which sits on top of [Archetypal](https://github.com/samuelduchesne/archetypal) and [eppy](https://github.com/santoshphilip/eppy).

This repository is under very active development as of October 2024.
