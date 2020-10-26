using Newtonsoft.Json;
using System;

namespace LoveSense.Domaine
{
    public class ResponseVerify: ResponseBase
    {
        [JsonProperty("verdict")]
        public bool Verdict { get; set; }

        [JsonProperty("score")]
        public decimal Score { get; set; }
    }
}
