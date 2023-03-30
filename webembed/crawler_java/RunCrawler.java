import com.crawljax.browser.EmbeddedBrowser;
import com.crawljax.core.CrawljaxRunner;
import com.crawljax.core.configuration.BrowserConfiguration;
import com.crawljax.core.configuration.CrawlRules;
import com.crawljax.core.configuration.CrawljaxConfiguration;
import com.crawljax.plugins.crawloverview.CrawlOverview;
import state_abstraction_function.Word2VecEmbeddingStateVertexFactory;

import java.util.concurrent.TimeUnit;

public class RunCrawler {
    private static final long WAIT_TIME_AFTER_EVENT = 500;
    private static final long WAIT_TIME_AFTER_RELOAD = 500;
    private static final String URL = "https://www.york.ac.uk/teaching/cws/wws/webpage1.html";

    public static void main(String[] args) throws Exception {

        CrawljaxConfiguration.CrawljaxConfigurationBuilder builder = CrawljaxConfiguration.builderFor(URL);
//      1. set crawl rules
        builder.crawlRules().setFormFillMode(CrawlRules.FormFillMode.RANDOM);
        builder.crawlRules().clickDefaultElements();
        builder.crawlRules().crawlHiddenAnchors(true);
        builder.crawlRules().crawlFrames(false);
        builder.crawlRules().clickElementsInRandomOrder(false);
//      2. set max number of states
//        builder.setMaximumStates(maxStates);
        builder.setUnlimitedStates();
//      3. set max run time
        builder.setMaximumRunTime(3, TimeUnit.MINUTES);
//        builder.setUnlimitedRuntime();
//      4. set crawl depth
        builder.setUnlimitedCrawlDepth();
//      5. setup abstract function to be used
        builder.setStateVertexFactory(new Word2VecEmbeddingStateVertexFactory());
//      6. set timeouts
        builder.crawlRules().waitAfterReloadUrl(WAIT_TIME_AFTER_RELOAD, TimeUnit.MILLISECONDS);
        builder.crawlRules().waitAfterEvent(WAIT_TIME_AFTER_EVENT, TimeUnit.MILLISECONDS);
//      7. choose browser
        builder.setBrowserConfig(new BrowserConfiguration(EmbeddedBrowser.BrowserType.CHROME, 1));
//      8. add crawl overview
        builder.addPlugin(new CrawlOverview());
//      9. build crawler
        CrawljaxRunner crawljax = new CrawljaxRunner(builder.build());
//      10. run crawler
        crawljax.call();


    }
}
