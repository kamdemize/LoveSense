using LoveSense.Domaine;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace LoveSense.Service
{
    public class CorpusExtractor : ICorpusExtractor
    {
        public async Task<IEnumerable<DocumentCorpus>> GetCorpusAsync()
        {
            HttpResponseMessage response = await HttpClientFacrory.Instance.Get().GetAsync("api/corpus");
            if (response != null && response.IsSuccessStatusCode)
            {
                var corpus = await response.Content.ReadAsAsync<IEnumerable<DocumentCorpus>>();
                return corpus;
            }

            return null;
        }
    }
}








