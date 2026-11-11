# External backbone source and paper registry

| Model | Original paper | Official source | Immutable commit | Entry / SHA256 | License |
|---|---|---|---|---|---|
| NLINEAR | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [official](https://github.com/cure-lab/LTSF-Linear) | 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 | models/NLinear.py / cabdcf5ac2658faa45b8887846182d222eb0a98c42059079c533df0fe689d879 | Apache-2.0 |
| DLINEAR | [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | [official](https://github.com/cure-lab/LTSF-Linear) | 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 | models/DLinear.py / 0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc | Apache-2.0 |
| FITS | [FITS: Modeling Time Series with 10k parameters](https://arxiv.org/abs/2307.03756) | [official](https://github.com/VEWOXIC/FITS) | d040bb015b6299da26d879b90dd19c80fb72c160 | models/FITS.py / 11c28af651bada1ba3e1e19c1cec1e835cc4b974d75da27df7b84d322096c90c | Apache-2.0 |
| PATCHTST | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) | [official](https://github.com/yuqinie98/PatchTST) | 204c21efe0b39603ad6e2ca640ef5896646ab1a9 | PatchTST_supervised/models/PatchTST.py / 49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b | Apache-2.0 |
| ITRANSFORMER | [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625) | [official](https://github.com/thuml/iTransformer) | c2426e68ca13f74aaec08045c5c724d8ad328124 | model/iTransformer.py / 6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db | MIT |
| SIGMA | [Generalizing Multi-Scale Time-Series Modeling with a Single Operator](https://proceedings.mlr.press/v306/lee26al.html) | [official](https://github.com/cheonwoolee/SiGMA) | b4ac7c967f4a873c6908978230485125f71f6419 | models/SiGMA.py / 409627a8005f39cef59bc78f4932fb4cca49e4b7a403285be4a4c904fe06e213 | MIT |
| TIMEMIXER | [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://arxiv.org/abs/2405.14616) | [official](https://github.com/kwuking/TimeMixer) | e24610583b36fdd8c76cc17a8df4e65759a5f460 | models/TimeMixer.py / 817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be | Apache-2.0 |
| OLINEAR | [OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain](https://arxiv.org/abs/2505.08550) | [official](https://github.com/jackyue1994/OLinear) | f168e01a3e0e316ad98330b5e77afed1f77b0af5 | model/OLinear.py / d7b08d7e1a09a732adef03ed40fe956d284a6b5f221a3be8e11bc018bb48b457 | NO_EXPLICIT_LICENSE_FOUND |

Recursive pinned file hashes and upstream license-file locations are in configuration/sources.json. All eight implementations are external;none is redistributed here.

## PostY output adapter

The DLinear comparison uses the official offline multiplicative PostY module from the pinned [Anoise/Adapter snapshot](https://github.com/Anoise/Adapter/tree/0add06ea7b4d2e0a84c364a8be72eef2676a92f2). No explicit license file was found in the inspected snapshot. Obtain the source separately under its applicable terms. The [configuration and source hashes](../supplementary/posty/protocol.json) identify the module and scheduler used in the experiment.
