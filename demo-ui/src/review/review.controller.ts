import { Controller, Get } from "@nestjs/common";
import { ReviewService } from "./review.service";

@Controller("api")
export class ReviewController {
  constructor(private readonly reviewService: ReviewService) {}

  @Get("pr-review")
  getPrReview() {
    return this.reviewService.getPrReview();
  }

  @Get("drift")
  getDriftReport() {
    return this.reviewService.getDriftReport();
  }

  @Get("root-cause")
  getRootCause() {
    return this.reviewService.getRootCause();
  }
}
