using Newtonsoft.Json;

namespace LoveSense.Domaine
{
    public class ResponseBase
    {
        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("msg")]
        public string Msg { get; set; }
    }
}
