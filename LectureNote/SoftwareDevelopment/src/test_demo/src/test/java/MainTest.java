import static org.junit.Assert.assertTrue;
import org.junit.Test;

public class MainTest {
	public static void main(String[] args) {
		System.err.println("Run the tests");
	}

	@Test
	public void test1() {
		boolean res = Example.example(0, 0);
		assertTrue(res);
	}

	@Test
	public void test2() {
		boolean res = Example.example(1, 10);
		assertTrue(!res);
	}
	
	@Test
	public void test3() {
		boolean res = Example.example(-3, 5);
		assertTrue(res);
	}
}
