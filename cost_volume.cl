//WTA for LF
__kernel void winnerTakesAll_LF(__global float * datacost, __global uchar * dispFinal, int dispRange, float dispScale)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);

	float minCost = FLT_MAX;
	float disparity = 0;
	for (int d = 0; d < dispRange; d++)
	{
		float cost = datacost[y + x * height + width * height * (d)];
		if (minCost > cost)
		{
			minCost = cost;
			disparity = (float)d;
		}
	}
	//disparity.w = 255;
	dispFinal[y + x * height] =  255 - (uchar)(disparity * dispScale);
	//dispFinal[y + x * height] = (disparity);
}