package de.hanslovsky.examples;

import ij.ImageJ;
import net.imglib2.algorithm.gradient.PartialDerivative;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

public class PartialDerivative3D
{

	public static void main( final String[] args )
	{
		new ImageJ();
		final ArrayImg< DoubleType, DoubleArray > img = ArrayImgs.doubles( 3, 3, 3 );
		final ArrayRandomAccess< DoubleType > ra = img.randomAccess();
		ra.setPosition( new int[] { 1, 1, 1 } );
		ra.get().set( 4.0 );
		ImageJFunctions.show( img );
		final ArrayImg< DoubleType, DoubleArray > grad1 = ArrayImgs.doubles( 3, 3, 3 );
		final ArrayImg< DoubleType, DoubleArray > grad2 = ArrayImgs.doubles( 3, 3, 3 );
		PartialDerivative.gradientCentralDifference( Views.extendBorder( img ), grad1, 1 );
		ImageJFunctions.show( grad1 );
		PartialDerivative.gradientCentralDifference( Views.extendBorder( grad1 ), grad2, 2 );
		ImageJFunctions.show( grad2 );
	}

}
