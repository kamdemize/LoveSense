using Newtonsoft.Json;
using System;

namespace LoveSense.Domaine
{
    public partial class SessionVerification: ResponseBase
    {
        [JsonProperty("date_session")]
        public DateTime DateSession { get; set; }

        [JsonProperty("verdict")]
        public bool Verdict { get; set; }

        [JsonProperty("text")]
        public string Text { get; set; }

        [JsonProperty("score")]
        public decimal Score { get; set; }
    }
}
