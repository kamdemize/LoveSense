using LoveSense.Domaine;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace LoveSense.Service
{
    public class SessionVerificator : ISessionVerificator
    {
        public async Task<IEnumerable<SessionVerification>> GetSessionsAsync()
        {
            HttpResponseMessage response = await HttpClientFacrory.Instance.Get().GetAsync("api/sessionverification");
            if (response != null && response.IsSuccessStatusCode)
            {
                var Sessions = await response.Content.ReadAsAsync<IEnumerable<SessionVerification>>();
                return Sessions;
            }

            return null;
        }
    }
}
