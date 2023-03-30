package state_abstraction_function;

import com.crawljax.core.state.StateVertexImpl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class Word2VecEmbeddingStateVertexImpl extends StateVertexImpl {
    private String classifierURI;

    {
        classifierURI = null;
    }

    private static URL restUrl;
    private static HttpURLConnection connection;

    /**
     * Defines a State.
     *
     * @param id
     * @param url         the current url of the state
     * @param name        the name of the state
     * @param dom         the current DOM tree of the browser
     * @param strippedDom the stripped dom by the OracleComparators
     */
    public Word2VecEmbeddingStateVertexImpl(int id, String url, String name, String dom, String strippedDom, String classifierURI) {
        super(id, url, name, dom, strippedDom);

        this.classifierURI = classifierURI;
    }
//  TODO finish implementing abstract function

    @Override
//    compares two pages and makes an HTTP request to the exposed python app
    public boolean equals(Object object) {
        try {
            Word2VecEmbeddingStateVertexImpl that = (Word2VecEmbeddingStateVertexImpl) object;

            String this_dom = this.getDom();
            String this_URL = this.getUrl();
            String this_strippedDom = this.getStrippedDom();

            String that_dom = that.getDom();
            String that_URL = that.getUrl();
            String that_strippedDom = that.getStrippedDom();

            // set up connection to flask
            restUrl = new URL("http://127.0.0.1:5000/equals");
            connection = (HttpURLConnection) restUrl.openConnection();

            // POST to pass params to python function
            connection.setRequestMethod("POST");

            connection.setRequestProperty("Content-Type", "application/json; utf-8");
            connection.setDoOutput(true);

            // TODO serialize object in some way and create json
            // dummy json
            String jsonInputString = "{\"dom1\": \"" + this_dom + "\",
                                       \"dom2\": \"" + that_dom + "\"}";

            try(OutputStream os = connection.getOutputStream()) {
                byte[] input = jsonInputString.getBytes("utf-8");
                os.write(input, 0, input.length);
            }

            BufferedReader reader =  new BufferedReader(new InputStreamReader(connection.getInputStream()));
            StringBuffer responseContent = new StringBuffer();
            String line;

            int status = connection.getResponseCode();

            if (status == 200){
                while ((line = reader.readLine()) != null){
                    responseContent.append(line);
                }
                reader.close();
            }

            boolean result = responseContent.toString().equals("true");

            System.out.println("Result is " + result);
            return result;

        } catch (IOException e){
            e.printStackTrace();
        }
        return false;
    }
}
