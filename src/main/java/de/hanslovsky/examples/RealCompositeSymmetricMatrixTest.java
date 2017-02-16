package de.hanslovsky.examples;

import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

import net.imglib2.algorithm.linalg.matrix.RealCompositeSymmetricMatrix;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;
import net.imglib2.view.composite.RealComposite;

public class RealCompositeSymmetricMatrixTest
{
	public static void main( final String[] args )
	{
		final ArrayImg< DoubleType, DoubleArray > img = ArrayImgs.doubles( new double[] { 1, 2, 3, 4, 5, 6 }, 1 );
		final RealComposite< DoubleType > comp = Views.collapseReal( img ).randomAccess().get();
		final RealCompositeSymmetricMatrix< DoubleType > matrix = new RealCompositeSymmetricMatrix< DoubleType >( comp, 3, 6 );
		System.out.println( matrix.toString() );

		System.out.println();

		final RealMatrix matrix2 = matrix.copy();
		matrix2.setEntry( 1, 2, 12 );
		System.out.println( matrix.toString() );
		System.out.println( matrix2.toString() );

		System.out.println();

		final RealMatrix newMat = matrix.createMatrix( 3, 3 );
		System.out.println( newMat );
		final Random rng = new Random( 100 );
		for ( int r = 0; r < newMat.getRowDimension(); ++r )
			for ( int c = r; c < newMat.getColumnDimension(); ++c )
			{
				newMat.setEntry( r, c, rng.nextInt( 10000 ) );
				System.out.println( newMat );
			}
	}
}
