using LoveSense.Domaine;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace LoveSense.Service
{
    public class ExperienceModeler : IExperienceModeler
    {
        public async Task<IEnumerable<MLExperience>> GetMLExperiencesAsync()
        {
            HttpResponseMessage response = await HttpClientFacrory.Instance.Get().GetAsync("api/mlexperience");
            if (response != null && response.IsSuccessStatusCode)
            {
                var mlEperiences = await response.Content.ReadAsAsync<IEnumerable<MLExperience>>();
                return mlEperiences;
            }

            return null;
        }
    }
}
