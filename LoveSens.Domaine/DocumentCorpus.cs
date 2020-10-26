using Newtonsoft.Json;
using System;

namespace LoveSense.Domaine
{
    public class DocumentCorpus: ResponseBase
    {
        [JsonProperty("date_creation")]
        public DateTime DateCreation { get; set; }

        [JsonProperty("label")]
        public string Label { get; set; }

        [JsonProperty("text")]
        public string Text { get; set; }
    }
}
