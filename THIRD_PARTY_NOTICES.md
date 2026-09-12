# Third-party notices

## Apple CoreNet / ByteFormer

`byteformer_model.py` is a modified, dependency-light port of Apple's CoreNet
ByteFormer Tiny implementation. It retains the full 12-layer Tiny backbone,
all 167 pretrained backbone tensors, and the official checkpoint's parameter
names and shapes. No affiliation with or endorsement by Apple is claimed.

Original source: https://github.com/apple/corenet

Files used:

- `corenet/modeling/models/classification/byteformer.py`
- `corenet/modeling/models/classification/config/byteformer.py`
- `corenet/modeling/modules/windowed_transformer.py`
- `corenet/modeling/modules/transformer.py`
- `corenet/modeling/layers/multi_head_attention.py`
- `corenet/modeling/layers/token_merging.py`
- `corenet/modeling/layers/positional_embedding.py`
- `corenet/modeling/layers/normalization/layer_norm.py`

Paper: *Bytes Are All You Need: Transformers Operating Directly On File Bytes*,
https://arxiv.org/abs/2306.00238

Checkpoint/config: ImageNet JPEG quality 100, Tiny, kernel 8, window 128;
https://github.com/apple/corenet/tree/main/projects/byteformer

### Changes made for the course

- Replace the training framework, registry, and custom basic layers with plain
  PyTorch; retain the original architecture and all backbone weights.
- Replace the 1000-class ImageNet classifier with a trainable 10-class MNIST
  classifier, with strict validation of every backbone key and shape.
- Actually apply the input padding mask. In the upstream source consulted on
  2026-09-12, `mask[x == -1].fill_(-inf)` writes into an indexed copy.
- Actually pass the combined padding/shift mask to window attention. The
  upstream source computes `total_mask` but passes the original `attn_mask`.
- Only full valid convolution windows are pooled. This ensures additional right
  padding does not introduce partially padded tokens absent from an unpadded
  forward pass. Merge masks retain partially valid pairs as in the original.
- Use unambiguous channel-last layer normalization, and avoid mutating the
  caller's byte tensor. The upstream normalization guesses channel layout from
  dimensions and is ambiguous when sequence length equals embedding dimension.
- Use PyTorch scaled-dot-product attention by default for efficient training.
  Manual attention is retained for exact reference comparison.

These corrections deliberately change upstream logits. Official forward parity
is tested only with `corrected_masks=False, use_sdpa=False`; this compatibility
mode reproduces the original behavior and is not the course training default.
Padding invariance and finite backbone gradients are independently tested for
the corrected course mode. `verify_model.py` produces the validation report.

### Apple source license (verbatim)

Copyright (C) 2024 Apple Inc. All Rights Reserved.

Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple
Inc. ("Apple") in consideration of your agreement to the following
terms, and your use, installation, modification or redistribution of
this Apple software constitutes acceptance of these terms.  If you do
not agree with these terms, please do not use, install, modify or
redistribute this Apple software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, Apple grants you a personal, non-exclusive
license, under Apple's copyrights in this original Apple software (the
"Apple Software"), to use, reproduce, modify and redistribute the Apple
Software, with or without modifications, in source and/or binary forms;
provided that if you redistribute the Apple Software in its entirety and
without modifications, you must retain this notice and the following
text and disclaimers in all such redistributions of the Apple Software.
Neither the name, trademarks, service marks or logos of Apple Inc. may
be used to endorse or promote products derived from the Apple Software
without specific prior written permission from Apple.  Except as
expressly stated in this notice, no other rights or licenses, express or
implied, are granted by Apple herein, including but not limited to any
patent rights that may be infringed by your derivative works or by other
works in which the Apple Software may be incorporated.

The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.

IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


-------------------------------------------------------------------------------
SOFTWARE DISTRIBUTED IN THIS REPOSITORY:

This software includes a number of subcomponents with separate
copyright notices and license terms - please see the file ACKNOWLEDGEMENTS.
-------------------------------------------------------------------------------


## MNIST data

MNIST was created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges.
Original dataset information: https://yann.lecun.org/exdb/mnist/

The data download location or mirror does not confer authorship on this course.
Follow the download source's dataset terms. The course repository's code license
must not be interpreted as a new license for MNIST or Apple's pretrained weights.
