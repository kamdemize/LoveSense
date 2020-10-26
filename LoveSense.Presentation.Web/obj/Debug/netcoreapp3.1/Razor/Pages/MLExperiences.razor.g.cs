#pragma checksum "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "e5cb15f5648ba33ddd24c8eafc304fdb06867691"
// <auto-generated/>
#pragma warning disable 1591
namespace LoveSense.Presentation.Web.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\_Imports.razor"
using LoveSense.Presentation.Web.Shared;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
using LoveSense.Service;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
using LoveSense.Presentation.Web.Models;

#line default
#line hidden
#nullable disable
    [Microsoft.AspNetCore.Components.RouteAttribute("/mlexperience")]
    public partial class MLExperiences : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.AddMarkupContent(0, "<h1>Machine learning experiences</h1>\r\n\r\n");
#nullable restore
#line 9 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
 if (mlExperiences == null)
{

#line default
#line hidden
#nullable disable
            __builder.AddContent(1, "    ");
            __builder.AddMarkupContent(2, "<div class=\"text-center\">\r\n        <div class=\"spinner-border m-5\" role=\"status\">\r\n            <span class=\"sr-only\">Loading...</span>\r\n        </div>\r\n    </div>\r\n");
#nullable restore
#line 16 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
}
else
{

#line default
#line hidden
#nullable disable
            __builder.AddContent(3, "    ");
            __builder.OpenElement(4, "table");
            __builder.AddAttribute(5, "class", "table");
            __builder.AddMarkupContent(6, "\r\n        ");
            __builder.AddMarkupContent(7, @"<thead>
            <tr>
                <th>Date</th>
                <th>Code</th>
                <th>Type</th>
                <th>Score</th>
                <th>Training Time</th>
                <th>Test Time</th>
                <th>Error</th>
            </tr>
        </thead>
        ");
            __builder.OpenElement(8, "tbody");
            __builder.AddMarkupContent(9, "\r\n");
#nullable restore
#line 32 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
             foreach (var mlExperience in mlExperiences)
            {

#line default
#line hidden
#nullable disable
            __builder.AddContent(10, "                ");
            __builder.OpenElement(11, "tr");
            __builder.AddMarkupContent(12, "\r\n                    ");
            __builder.OpenElement(13, "td");
            __builder.AddContent(14, 
#nullable restore
#line 35 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.DateExperience.ToString("dd/MM/yyyy HH:mm")

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(15, "\r\n                    ");
            __builder.OpenElement(16, "td");
            __builder.AddContent(17, 
#nullable restore
#line 36 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.Code

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(18, "\r\n                    ");
            __builder.OpenElement(19, "td");
            __builder.AddContent(20, 
#nullable restore
#line 37 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.ExperienceType

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(21, "\r\n                    ");
            __builder.OpenElement(22, "td");
            __builder.AddContent(23, 
#nullable restore
#line 38 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.Score.ToString("P")

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(24, "\r\n                    ");
            __builder.OpenElement(25, "td");
            __builder.AddContent(26, 
#nullable restore
#line 39 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.TrainingTime.ToString("F")

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(27, "\r\n                    ");
            __builder.OpenElement(28, "td");
            __builder.AddContent(29, 
#nullable restore
#line 40 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.TestTime.ToString("F")

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(30, "\r\n                    ");
            __builder.OpenElement(31, "td");
            __builder.AddContent(32, 
#nullable restore
#line 41 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
                         mlExperience.Error

#line default
#line hidden
#nullable disable
            );
            __builder.CloseElement();
            __builder.AddMarkupContent(33, "\r\n                ");
            __builder.CloseElement();
            __builder.AddMarkupContent(34, "\r\n");
#nullable restore
#line 43 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
            }

#line default
#line hidden
#nullable disable
            __builder.AddContent(35, "        ");
            __builder.CloseElement();
            __builder.AddMarkupContent(36, "\r\n    ");
            __builder.CloseElement();
            __builder.AddMarkupContent(37, "\r\n");
#nullable restore
#line 46 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
}

#line default
#line hidden
#nullable disable
        }
        #pragma warning restore 1998
#nullable restore
#line 48 "D:\Activities\Teluq\MTI6012\Dev\LoveSens\LoveSense.Presentation.Web\Pages\MLExperiences.razor"
       
    private IEnumerable<MLExperienceModel> mlExperiences;

    protected override async Task OnInitializedAsync()
    {
        var mlExperiencesAsync = await ExperienceModeler.GetMLExperiencesAsync();
        mlExperiences = mlExperiencesAsync?.Select(x => new MLExperienceModel
        {
            Code = x.Code,
            DateExperience = x.DateExperience,
            ExperienceType = x.ExperienceType.ToString(),
            Score = x.Score,
            TrainingTime = x.TrainingTime,
            TestTime = x.TestTime,
            Error = x.Error,
        });
    }

#line default
#line hidden
#nullable disable
        [global::Microsoft.AspNetCore.Components.InjectAttribute] private IExperienceModeler ExperienceModeler { get; set; }
    }
}
#pragma warning restore 1591