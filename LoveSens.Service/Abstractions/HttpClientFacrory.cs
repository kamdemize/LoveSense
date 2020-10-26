using System;
using System.Net.Http;
using System.Net.Http.Headers;

namespace LoveSense.Service
{
    public sealed class HttpClientFacrory
    {
        private HttpClientFacrory()
        {
        }
        private static readonly Lazy<HttpClientFacrory> lazy = new Lazy<HttpClientFacrory>(() => new HttpClientFacrory());
        public static HttpClientFacrory Instance
        {
            get
            {
                return lazy.Value;
            }
        }

        public HttpClient Get()
        {
            var client = new HttpClient
            {
                BaseAddress = new Uri("http://localhost:5000/"),
            };
            client.DefaultRequestHeaders.Accept.Clear();
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));

            return client;
        }
    }
}
