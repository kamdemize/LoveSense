using System;
using System.Threading.Tasks;
using System.Net.Http;
using System.Net.Http.Headers;
using LoveSense.Domaine;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

namespace LoveSense.Service
{
    public class MessageVerificator : IMessageVerificator
    {
        public async Task<ResponseVerify> VerifyAsync(string text)
        {
            string message = "{ text: \""+ text + "\"}";
            var messageJsonify = JObject.Parse(message);
            HttpResponseMessage response = await HttpClientFacrory.Instance.Get().PostAsJsonAsync("api/message/Verify", messageJsonify);
            if (response != null && response.IsSuccessStatusCode)
            {
                var jsonString = await response.Content.ReadAsStringAsync();
                var responseData = JsonConvert.DeserializeObject<ResponseVerify>(jsonString);
                return responseData;
            }

            return null;
        }
    }
}
