# Third-party notices

## LarsonLab Spectral-Spatial RF Pulse Design

The native Python SPSP implementation in this project is based on the published
Larson/Kerr method and was developed by studying the public MATLAB reference
toolbox:

<https://github.com/LarsonLab/Spectral-Spatial-RF-Pulse-Design>

No MATLAB files or compiled binaries are distributed with this project. The
Python implementation does not call MATLAB and does not incorporate source from
the toolbox's `rf_tools` directory.

The repository-level `LICENSE` file of the reference toolbox contains the
following BSD-3-Clause notice:

> Copyright (c) 2011-2015, Board of Trustees, Leland Stanford Junior University
> and The Regents of the University of California.
>
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> - Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> - Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> - Neither the name of Spectral-Spatial-RF-Pulse-Design nor the names of its
>   contributors may be used to endorse or promote products derived from this
>   software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

### Legacy notice and `rf_tools`

The same reference repository also contains a legacy `Copyright_Information`
file with educational, research, and non-profit wording, while the current
top-level `LICENSE` is BSD-3-Clause. The embedded `rf_tools/README` identifies
the package as John Pauly's MATLAB RF tools but does not state a separate
license. Because the provenance and licensing scope of those embedded files is
not unambiguous, no `rf_tools` source, MEX source, binary, or direct translation
was incorporated. SciPy's independently implemented signal-processing
functions are used instead.

The attribution in this notice and in the GUI is informational and does not
imply endorsement by LarsonLab, Stanford University, the University of
California, or any contributor.
