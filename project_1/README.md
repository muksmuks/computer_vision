What are Channels and Kernels (according to EVA)?
==================================================================================================================
Channels refer to the number of colors which represents an image. For example, there are three channels in a RGB image, the Red Channel, the Green Channel and the Blue Channel. Each of the channels in each pixel represents the intensity of each color that constitute that pixel.

An an image, we are essentially seeing lots of distinguishable features together. Kernel is a feature extractor. It finds a feature wherever it occurs in an image.

Why should we only (well mostly) use 3x3 Kernels?
==================================================================================================================

Because GPU are optimised to handle it. Further any other filters(7x7) can be expressed in terms of multiple 3x3

Why not even filters. As we need determining edges. A even number filter cannot express edge

How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
==================================================================================================================
199|197|195|193|191|189|187|185|183|181|179|177|175|173|171|169|167|165|163|161|159|157|155|153|151|149|147|145|143|141|139|137|135|133|131|129|127|125|123|121|119|117|115|113|111|109|107|105|103|101|99|97|95|93|91|89|87|85|83|81|79|77|75|73|71|69|67|65|63|61|59|57|55|53|51|49|47|45|43|41|39|37|35|33|31|29|27|25|23|21|19|17|15|13|11|9|7|5|3|1

100 times



