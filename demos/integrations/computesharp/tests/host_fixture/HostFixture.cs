using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HostFixture
{
    // A host-side test project: it compiles no shader descriptor into its assembly.
    [TestClass]
    public sealed class Cases
    {
        [TestMethod]
        public void Plain() { }

        // MSTest names these rows with their string arguments in quotes.
        [DataTestMethod]
        [DataRow("checked")]
        [DataRow("two words")]
        public void Named(string text) { }

        [DataTestMethod]
        [DataRow("first", 1)]
        [DataRow("second", 2)]
        public void Mixed(string text, int value) { }

        [DataTestMethod]
        [DataRow("custom", DisplayName = "Displayed custom")]
        public void Displayed(string text) { }
    }
}
