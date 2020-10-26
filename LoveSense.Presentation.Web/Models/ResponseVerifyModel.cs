using Newtonsoft.Json;
using System;

namespace LoveSense.Presentation.Web.Models
{
    public class ResponseVerifyModel   
    {
        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("verdict")]
        public bool Verdict { get; set; }

        [JsonProperty("msg")]
        public string Message { get; set; }

        [JsonProperty("score")]
        public decimal Score { get; set; }
    }
}
