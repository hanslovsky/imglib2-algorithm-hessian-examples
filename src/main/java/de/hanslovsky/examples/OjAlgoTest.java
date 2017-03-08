package de.hanslovsky.examples;

import org.ojalgo.array.Array1D;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.scalar.ComplexNumber;

public class OjAlgoTest
{

	public static void main( final String[] args )
	{

		final PrimitiveDenseStore m = PrimitiveDenseStore.FACTORY.rows( new double[][] {
			{ 1, 2, 3 },
			{ 4, 5, 6 },
			{ 7, 8, 9 }
		} );

		System.out.println( m );

		final Eigenvalue< Double > ev = Eigenvalue.PRIMITIVE.make();
		Eigenvalue.PRIMITIVE.make( m, false );
		Array1D.COMPLEX.makeZero( 1 );


		ev.computeValuesOnly( m );
		final Array1D< ComplexNumber > e = ev.getEigenvalues();
		System.out.println( e );

	}

}
